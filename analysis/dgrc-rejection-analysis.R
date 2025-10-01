library(tidyverse)
library(lmerTest)

model_meta <- tribble(
  ~model, ~short, ~base, ~class, ~instruct, ~params,
  "llama-3-8b", "L-3-8B", "llama-3-8b", "Llama-3-8B", FALSE, 8000000000,
  "llama-3-8b-instruct", "L-3-8B-I", "llama-3-8b", "Llama-3-8B", TRUE, 8000000000,
  "qwen2.5-500m", "Q-2.5-500M", "qwen-2.5-500M", "Qwen2.5", FALSE, 500000000,
  "qwen2.5-500m-instruct", "Q-2.5-500M-I", "qwen-2.5-500M", "Qwen2.5", TRUE, 500000000,
  "qwen2.5-1.5b", "Q-2.5-1.5B", "qwen-2.5-1.5B", "Qwen2.5", FALSE, 1500000000,
  "qwen2.5-1.5b-instruct", "Q-2.5-1.5B-I", "qwen-2.5-1.5B", "Qwen2.5", TRUE, 1500000000,
  "qwen2.5-3b", "Q-2.5-3B", "qwen-2.5-3B", "Qwen2.5", FALSE, 3000000000,
  "qwen2.5-3b-instruct", "Q-2.5-3B-I", "qwen-2.5-3B", "Qwen2.5", TRUE, 3000000000,
  "qwen2.5-7b", "Q-2.5-7B", "qwen-2.5-7B", "Qwen2.5", FALSE, 7000000000,
  "qwen2.5-7b-instruct", "Q-2.5-7B-I", "qwen-2.5-7B","Qwen2.5", TRUE, 7000000000,
) %>%
  mutate(
    short = factor(short, levels = c("L-3-8B", "L-3-8B-I", "Q-2.5-500M", "Q-2.5-500M-I", 
                                     "Q-2.5-1.5B", "Q-2.5-1.5B-I", "Q-2.5-3B", "Q-2.5-3B-I", 
                                     "Q-2.5-7B", "Q-2.5-7B-I")),
    base = factor(base, levels = c("llama-3-8b", "qwen-2.5-500M", "qwen-2.5-1.5B", "qwen-2.5-3B", "qwen-2.5-7B"))
  )

stimuli <- fs::dir_ls("data/results/sorted-generations/rejection/", regexp = "*.csv") %>%
  map_df(read_csv, .id = "model") %>%
  group_by(model) %>%
  mutate(
    idx = row_number()
  ) %>%
  ungroup() %>%
  mutate(
    model = str_remove(model, "data/results/sorted-generations/rejection/"),
    model = str_remove(model, ".csv"),
    mode = case_when(
      str_detect(model, "arc") ~ "arc",
      TRUE ~ "coord"
    ),
    model = str_remove(model, "-(arc|coord)")
  )

results <- fs::dir_ls("data/results/dgrc/", recurse = TRUE, regexp = "*/rejection-(arc|coord)\\/*\\/") %>%
  keep(str_detect(., ".csv")) %>%
  map_df(read_csv, .id = "model") %>%
  group_by(model) %>%
  mutate(
    idx = row_number()
  ) %>%
  ungroup() %>%
  mutate(
    model = str_remove(model, "data/results/dgrc/rejection-"),
    model = str_remove(model, ".csv"),
    mode = case_when(
      str_detect(model, "arc") ~ "arc",
      TRUE ~ "coord"
    ),
    model = str_remove(model, "(arc|coord)/")
  ) %>%
  group_by(model, header, mode) %>%
  mutate(
    idx = row_number()
  ) %>%
  ungroup()


nested <- results %>%
  inner_join(stimuli) %>%
  select(-idx) %>%
  group_by(model, mode, swapped, item, header) %>%
  nest()

metric <- nested %>%
  mutate(
    recency = map_dbl(data, function(item) {
      vp1 <- item %>%
        filter(continuation_type == "vp1") %>%
        pull(score)
      
      vp2 <- item %>%
        filter(continuation_type == "vp2") %>%
        pull(score)
      
      expand_grid(vp1, vp2) %>%
        summarize(
          prop = mean(vp2 > vp1)
        ) %>%
        pull(prop) %>%
        .[1]
    })
  ) %>%
  ungroup() %>%
  mutate(
    recency = case_when(
      swapped == TRUE ~ 1-recency,
      TRUE ~ recency
    )
  ) %>%
  select(-data)

metric

metric %>%
  group_by(model, mode, swapped, header) %>%
  summarize(
    n = n(),
    sd = sd(recency),
    cb = qt(0.05/2, n-1, lower.tail = FALSE) * sd/sqrt(n),
    mean = mean(recency)
  ) %>%
  ungroup() %>%
  filter(swapped == FALSE) %>%
  inner_join(model_meta) %>%
  mutate(
    mode = case_when(
      mode == "arc" ~ "ARC",
      TRUE ~ "Coord"
    ),
    header = case_when(
      header == "no" ~ "Absent",
      TRUE ~ "Present"
    )
  ) %>%
  ggplot(aes(params/1e9, mean, color = header, fill = header, shape = class, linetype = instruct)) +
  geom_point(size = 2.5) +
  geom_line() +
  # geom_ribbon(aes(ymin = mean-cb, ymax = mean+cb), color = NA, alpha = 0.2) +
  geom_linerange(aes(ymin = mean-cb, ymax = mean+cb), linetype = "solid", linewidth = 0.3) +
  geom_hline(yintercept = 0.5, linetype = "dashed") +
  scale_y_continuous(limits = c(0.3,0.7), labels = scales::percent_format()) +
  scale_x_log10(limits = c(0.5, 8), breaks = c(0.5,1,2,4,6,8), labels = c("1/2", "1", "2", "4", "6", "8")) +
  # scale_color_brewer(palette = "Dark2", aesthetics = c("color", "fill")) +
  scale_color_manual(values = c("#7570b3", "#e6ab02"), aesthetics = c("color", "fill")) +
  # facet_grid(header ~ mode) +
  facet_wrap(~mode) +
  theme_bw(base_size = 16, base_family = "Times") +
  theme(
    axis.text = element_text(color = "black")
  ) +
  labs(
    x = "Parameters (in billion)",
    y = "VP2-preference",
    linetype = "Instruct",
    shape = "Model",
    color = "Digression Signal",
    fill = "Digression Signal"
  )

ggsave("plots/dgrc-digression-exp.pdf", width = 7.14, height = 3.60, dpi=300, device=cairo_pdf)


metric_reg <- metric %>%
  filter(swapped == FALSE) %>%
  inner_join(model_meta) %>%
  mutate(
    header = case_when(
      header == "no" ~ 0.5,
      TRUE ~ -0.5
    ),
    mode = case_when(
      mode == "coord" ~ -0.5,
      TRUE ~ 0.5
    ),
    instruct = case_when(
      instruct == TRUE ~ 0.5,
      TRUE ~ -0.5
    )
  )

fit2 <- lmer(recency ~ header * instruct * mode + (1 + header + instruct + mode | base) + (1 + header + instruct + mode | item), data= metric_reg)
summary(fit2)

emmip(fit2, header ~ mode, CIs = TRUE, plotit = FALSE) %>%
  as_tibble() %>%
  mutate(
    header = case_when(
      header == -0.5 ~ "Present",
      TRUE ~ "Absent"
    ),
    mode = case_when(
      mode == -0.5 ~ "Coord",
      TRUE ~ "ARC"
    )
  ) %>%
  ggplot(aes(mode, yvar, color = header, linetype = header, shape = header)) +
  geom_point(size = 2.5) +
  geom_line(aes(group = header)) +
  geom_linerange(linetype = "solid", aes(ymin = yvar-SE, ymax=yvar+SE)) +
  # scale_color_brewer(palette = "Dark2") +
  scale_color_manual(values = c("#7570b3", "#e6ab02")) +
  scale_y_continuous(labels = scales::percent_format(), limits = c(0.4,0.7)) +
  # guides(col = guide_legend(nrow = 2)) +
  theme_bw(base_size = 15, base_family = "Times") +
  theme(
    axis.text = element_text(color = "black"),
    panel.grid = element_blank(),
    legend.position = "top",
    legend.text = element_text(size = 12),
    legend.title = element_text(size = 13)
  ) +
  labs(
    x = "Structure",
    y = "Estimated VP2-Preference",
    color = "Digression Signal",
    fill = "Digression Signal",
    shape = "Digression Signal",
    linetype = "Digression Signal"
  )

# ggsave("plots/structure-digression-interaction.pdf", height = 3.5, width = 3.86, dpi = 300, device=cairo_pdf)
ggsave("plots/structure-digression-interaction.pdf", height = 4.04, width = 4.44, dpi = 400, device=cairo_pdf)


emmip(fit1.full, instruct ~ mode, CIs = TRUE, plotit = FALSE) %>%
  as_tibble() %>%
  mutate(
    instruct = case_when(
      instruct == -0.5 ~ FALSE,
      TRUE ~ TRUE
    ),
    mode = case_when(
      mode == -0.5 ~ "Coord",
      TRUE ~ "ARC"
    )
  ) %>%
  ggplot(aes(mode, yvar, color = instruct, linetype = instruct, shape = instruct)) +
  geom_point(size = 2.5) +
  geom_line(aes(group = instruct)) +
  geom_linerange(linetype = "solid", aes(ymin = yvar-SE, ymax=yvar+SE)) +
  scale_color_brewer(palette = "Dark2") +
  scale_y_continuous(labels = scales::percent_format(), limits = c(0.4,0.7)) +
  theme_bw(base_size = 16, base_family = "Times") +
  theme(
    axis.text = element_text(color = "black"),
    panel.grid = element_blank(),
    legend.position = "top"
  ) +
  labs(
    x = "Structure",
    y = "Estimated VP2-Preference",
    color = "Instruct",
    fill = "Instruct",
    shape = "Instruct",
    linetype = "Instruct"
  )

ggsave("plots/structure-instruct-interaction-digression-exp.pdf", height = 4.04, width = 4.44, dpi = 400, device=cairo_pdf)


# ggsave("plots/exp3-dgrc-rejection.pdf", width = 7.68, height = 6.47, dpi = 300, device=cairo_pdf)
# 768 x 647

ttested <- metric %>%
  pivot_wider(names_from = header, values_from = recency) %>%
  group_by(model, swapped, mode) %>%
  nest() %>%
  mutate(
    ttest = map(data, function(x) {
      t.test(x$no, x$wait, paired = TRUE) %>%
        broom::tidy()
    })
  ) %>%
  select(-data) %>%
  unnest(ttest) %>%
  ungroup() %>%
  inner_join(model_meta)

ttested %>%
  filter(swapped == FALSE) %>%
  ggplot(aes(params/1e9, estimate, color = instruct, fill = instruct, shape = class, linetype = instruct)) +
  geom_point(size = 2.5) +
  geom_line() +
  geom_linerange(aes(ymin = conf.low, ymax = conf.high), linetype = "solid") +
  geom_hline(yintercept = 0.0, linetype = "dashed") +
  # scale_y_continuous(limits = c(0,1), labels = scales::percent_format()) +
  scale_x_log10(limits = c(0.5, 8), breaks = c(0.5,1,2,4,6,8), labels = c("1/2", "1", "2", "4", "6", "8")) +
  scale_color_brewer(palette = "Dark2", aesthetics = c("color", "fill")) +
  # facet_grid(swapped~mode) +
  facet_wrap(~ mode) +
  theme_bw(base_size = 16) +
  theme(
    axis.text = element_text(color = "black")
  ) +
  labs(
    x = "Parameters (in billion)",
    y = "VP2 preference\ndiff (No - Wait)"
  )

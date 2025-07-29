library(tidyverse)

model_meta <- tribble(
  ~model, ~short, ~class, ~instruct, ~params,
  "llama-3-8b", "L-3-8B", "Llama-3-8B", FALSE, 8000000000,
  "llama-3-8b-instruct", "L-3-8B-I", "Llama-3-8B", TRUE, 8000000000,
  "qwen2.5-500m", "Q-2.5-500M", "Qwen2.5", FALSE, 500000000,
  "qwen2.5-500m-instruct", "Q-2.5-500M-I", "Qwen2.5", TRUE, 500000000,
  "qwen2.5-1.5b", "Q-2.5-1.5B", "Qwen2.5", FALSE, 1500000000,
  "qwen2.5-1.5b-instruct", "Q-2.5-1.5B-I", "Qwen2.5", TRUE, 1500000000,
  "qwen2.5-3b", "Q-2.5-3B", "Qwen2.5", FALSE, 3000000000,
  "qwen2.5-3b-instruct", "Q-2.5-3B-I", "Qwen2.5", TRUE, 3000000000,
  "qwen2.5-7b", "Q-2.5-7B", "Qwen2.5", FALSE, 7000000000,
  "qwen2.5-7b-instruct", "Q-2.5-7B-I", "Qwen2.5", TRUE, 7000000000,
) %>%
  mutate(
    short = factor(short, levels = c("L-3-8B", "L-3-8B-I", "Q-2.5-500M", "Q-2.5-500M-I", 
                                     "Q-2.5-1.5B", "Q-2.5-1.5B-I", "Q-2.5-3B", "Q-2.5-3B-I", 
                                     "Q-2.5-7B", "Q-2.5-7B-I")),
  )

stimuli <- fs::dir_ls("data/results/sorted-generations/freeform/", regexp = "*.csv") %>%
  map_df(read_csv, .id = "model") %>%
  group_by(model) %>%
  mutate(
    idx = row_number()
  ) %>%
  ungroup() %>%
  mutate(
    model = str_remove(model, "data/results/sorted-generations/freeform/"),
    model = str_remove(model, ".csv"),
    mode = case_when(
      str_detect(model, "arc") ~ "arc",
      TRUE ~ "coord"
    ),
    model = str_remove(model, "-(arc|coord)")
  )


results <- fs::dir_ls("data/results/dgrc/", recurse = TRUE, regexp = "*.csv") %>%
  map_df(read_csv, .id = "model") %>%
  group_by(model) %>%
  mutate(
    idx = row_number()
  ) %>%
  ungroup() %>%
  mutate(
    model = str_remove(model, "data/results/dgrc/freeform-"),
    model = str_remove(model, ".csv"),
    mode = case_when(
      str_detect(model, "arc") ~ "arc",
      TRUE ~ "coord"
    ),
    model = str_remove(model, "(arc|coord)/")
  )

nested <- results %>%
  inner_join(stimuli) %>%
  select(-idx) %>%
  group_by(model, mode, swapped, item) %>%
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

metric %>%
  group_by(model, mode, swapped) %>%
  summarize(
    n = n(),
    sd = sd(recency),
    cb = qt(0.05/2, n-1, lower.tail = FALSE) * sd/sqrt(n),
    mean = mean(recency)
  ) %>%
  ungroup() %>%
  inner_join(model_meta) %>%
  # filter(swapped == FALSE) %>%
  ggplot(aes(params/1e9, mean, color = instruct, fill = instruct, shape = class, linetype = swapped)) +
  geom_point(size = 2.5) +
  geom_line() +
  # geom_ribbon(aes(ymin = mean-cb, ymax = mean+cb), color = NA, alpha = 0.2) +
  geom_linerange(aes(ymin = mean-cb, ymax = mean+cb), linetype = "solid", linewidth = 0.3) +
  geom_hline(yintercept = 0.5, linetype = "dashed") +
  scale_y_continuous(limits = c(0,1), labels = scales::percent_format()) +
  scale_x_log10(limits = c(0.5, 8), breaks = c(0.5,1,2,4,6,8), labels = c("1/2", "1", "2", "4", "6", "8")) +
  scale_color_brewer(palette = "Dark2", aesthetics = c("color", "fill")) +
  facet_wrap(~mode) +
  theme_bw(base_size = 16) +
  theme(
    axis.text = element_text(color = "black")
  ) +
  labs(
    x = "Parameters (in billion)",
    y = "VP2-preference"
  )

ggsave("plots/exp1-dgrc.pdf", width = 7.68, height = 4.92, dpi=300, device=cairo_pdf)

# systematicity test

nested_sys <- results %>%
  inner_join(stimuli) %>%
  select(-idx) %>%
  group_by(model, mode, item) %>%
  nest()


metric_sys <- nested_sys %>%
  mutate(
    recency = map_dbl(data, function(item) {
      vp1_unswapped <- item %>%
        filter(continuation_type == "vp1" & swapped == FALSE) %>%
        pull(score)
      
      vp2_unswapped <- item %>%
        filter(continuation_type == "vp2"& swapped == FALSE) %>%
        pull(score)
      
      vp1_swapped <- item %>%
        filter(continuation_type == "vp1" & swapped == TRUE) %>%
        pull(score)
      
      vp2_swapped <- item %>%
        filter(continuation_type == "vp2"& swapped == TRUE) %>%
        pull(score)
      
      bind_cols(
        expand_grid(vp1_unswapped, vp2_unswapped),
        expand_grid(vp1_swapped, vp2_swapped)
      ) %>%
        summarize(
          prop = mean(vp2_unswapped > vp1_unswapped & vp1_swapped > vp2_swapped)
        ) %>%
        pull(prop) %>%
        .[1]
    })
  ) %>%
  ungroup() %>%
  select(-data)

metric_sys %>%
  group_by(model, mode) %>%
  summarize(
    n = n(),
    sd = sd(recency),
    cb = qt(0.05/2, n-1, lower.tail = FALSE) * sd/sqrt(n),
    mean = mean(recency)
  ) %>%
  ungroup() %>%
  inner_join(model_meta) %>%
  ggplot(aes(params/1e9, mean, color = instruct, fill = instruct, shape = class, linetype = instruct)) +
  geom_point(size = 2.5) +
  geom_line() +
  # geom_ribbon(aes(ymin = mean-cb, ymax = mean+cb), color = NA, alpha = 0.2) +
  geom_linerange(aes(ymin = mean-cb, ymax = mean+cb), linetype = "solid", linewidth = 0.3) +
  geom_hline(yintercept = 0.25, linetype = "dashed") +
  scale_y_continuous(limits = c(0,1), labels = scales::percent_format()) +
  scale_x_log10(limits = c(0.5, 8), breaks = c(0.5,1,2,4,6,8), labels = c("1/2", "1", "2", "4", "6", "8")) +
  scale_color_brewer(palette = "Dark2", aesthetics = c("color", "fill")) +
  facet_wrap(~mode) +
  theme_bw(base_size = 16) +
  theme(
    axis.text = element_text(color = "black")
  ) +
  labs(
    x = "Parameters (in billion)",
    y = "VP2-preference"
  )

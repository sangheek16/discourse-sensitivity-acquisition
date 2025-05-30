library(tidyverse)
library(fs)

model_meta <- tribble(
  ~model, ~short, ~class, ~instruct, ~params,
  "Meta-Llama-3-8B", "L-3-8B", "Llama-3-8B", FALSE, 8000000000,
  "Meta-Llama-3-8B-Instruct", "L-3-8B-I", "Llama-3-8B", TRUE, 8000000000,
  "Qwen2.5-0.5B", "Q-2.5-500M", "Qwen2.5", FALSE, 500000000,
  "Qwen2.5-0.5B-Instruct", "Q-2.5-500M-I", "Qwen2.5", TRUE, 500000000,
  "Qwen2.5-1.5B", "Q-2.5-1.5B", "Qwen2.5", FALSE, 1500000000,
  "Qwen2.5-1.5B-Instruct", "Q-2.5-1.5B-I", "Qwen2.5", TRUE, 1500000000,
  "Qwen2.5-3B", "Q-2.5-3B", "Qwen2.5", FALSE, 3000000000,
  "Qwen2.5-3B-Instruct", "Q-2.5-3B-I", "Qwen2.5", TRUE, 3000000000,
  "Qwen2.5-7B", "Q-2.5-7B", "Qwen2.5", FALSE, 7000000000,
  "Qwen2.5-7B-Instruct", "Q-2.5-7B-I", "Qwen2.5", TRUE, 7000000000,
) %>%
  mutate(
    short = factor(short, levels = c("L-3-8B", "L-3-8B-I", "Q-2.5-500M", "Q-2.5-500M-I", 
                                     "Q-2.5-1.5B", "Q-2.5-1.5B-I", "Q-2.5-3B", "Q-2.5-3B-I", 
                                     "Q-2.5-7B", "Q-2.5-7B-I")),
  )

kim22_arc <- read_csv("data/stimuli/kim22-arc.csv")

arc_results <- dir_ls("data/results/kim22-arc-metrics/", regexp = "*.csv") %>%
  map_df(read_csv, .id = "model") %>%
  mutate(
    model = str_extract(model, "(?<=kim22-arc-metrics/)(.*)(?=.csv)"),
    model = str_replace(model, "(meta-llama_|Qwen_)", ""),
    stimuli = "ARC"
  ) %>%
  mutate(
    # no_c1_relative = exp(no_prefix_c1)/(exp(no_prefix_c1) + exp(no_control)),
    # no_c2_relative = exp(no_prefix_c2)/(exp(no_prefix_c2) + exp(no_control)),
    # wait_c1_relative = exp(wait_prefix_c1)/(exp(wait_prefix_c1) + exp(wait_control)),
    # wait_c2_relative = exp(wait_prefix_c2)/(exp(wait_prefix_c2) + exp(wait_control)),
    no_c1_relative = exp(no_prefix_c1)/(exp(no_prefix_c1) + exp(no_prefix_c2)),
    no_c2_relative = exp(no_prefix_c2)/(exp(no_prefix_c2) + exp(no_prefix_c1)),
    wait_c1_relative = exp(wait_prefix_c1)/(exp(wait_prefix_c1) + exp(wait_prefix_c2)),
    wait_c2_relative = exp(wait_prefix_c2)/(exp(wait_prefix_c2) + exp(wait_prefix_c1)),
    # exp1 = (exp(wait_prefix_c1)/exp(wait_prefix_c2)) - (exp(no_prefix_c1)/exp(no_prefix_c2)),
    # exp1_alt = (wait_prefix_c1/wait_prefix_c2) - (no_prefix_c1/no_prefix_c2),
    # exp1_altalt = wait_c1_relative - no_c1_relative,
    exp1 = (wait_prefix_c1 - wait_prefix_c2) - (no_prefix_c1 - no_prefix_c2),
    exp1_2 = wait_prefix_c2 - wait_prefix_c1,
    # exp1 = wait_c1_relative - no_c1_relative,
    # exp2 = no_c2_relative  - no_c1_relative,
    # exp1 = wait_prefix_c1 - no_prefix_c1,
    # exp1 = wait_prefix_c1 - wait_prefix_c2,
    exp2 = no_prefix_c2 - no_prefix_c1,
  ) %>%
  inner_join(kim22_arc)

arc_results %>%
  select(idx, model, rejection_id, preamble, no, wait, continuation1, continuation2, wait_c1_relative, no_c1_relative,
         exp1, exp1_alt, exp1_altalt) %>% View()

arc_results %>% count(model)



exp1_relative <- arc_results %>%
  select(-name1, -name2) %>%
  select(model, idx, item, rejection_id, swapped, exp1:exp2) %>%
  pivot_longer(exp1:exp2, names_to = "exp", values_to = "score") %>%
  group_by(model, rejection_id, swapped, exp) %>%
  summarize(
    satisfied = mean(score > 0)
  ) %>%
  ungroup()

# exp1_relative %>%
#   pivot_wider(names_from = exp, values_from = satisfied) %>%
#   ggplot(aes(exp1, exp1_alt)) +
#   geom_point() + 
#   facet_wrap(~model)

rejection_combos <- kim22_arc %>% distinct(rejection_id, no, wait) %>% mutate(combo = glue::glue("{no}/{wait}"))
# rejection_labels <- rejection_combos$combo[1:3]
rejection_labels <- rejection_combos$combo
rejection_values <- c("\u0030", "\u0031", "\u0032", "\u0033", "\u0034", "\u0035", "\u0036", "\u0037", "\u0038", "\u0039", "\u0031\u0030", "\u0031\u0031", "\u0031\u0032", "\u0031\u0033", "\u0031\u0034", "\u0031\u0035")
# rejection_values <- as.character(Unicode::as.u_char(0:16)) %>% str_replace("U\\+", "\\u")


exp1_relative %>%
  # mutate(
  #   exp = case_when(
  #     exp == "exp1" ~ "Wait ARC\n>\nNo ARC",
  #     exp == "exp1_alt" ~ "Wait ARC\n>\nNo ARC",
  #     exp == "exp2" ~ "No MC\n>\nNo ARC"
  #   ),
  #   exp = factor(exp, levels = c("Wait ARC\n>\nNo ARC", "No MC\n>\nNo ARC")),
  # ) %>%
  filter(swapped == FALSE) %>%
  ggplot(aes(exp, satisfied, color = swapped)) +
  # geom_jitter(size = 2.5, width = 0.15, alpha = 0.6, seed = 1024) +
  geom_text(aes(label = rejection_id), position=position_jitter(width = 0.2, seed=1024), show.legend = TRUE) +
  geom_hline(yintercept = 0.5, linetype = "dashed") +
  scale_color_brewer(palette = "Dark2") +
  scale_y_continuous(limits = c(0,1), labels = scales::percent_format()) +
  # scale_shape_manual(
  #   name = "Rejection Combo",
  #   labels = rejection_labels,
  #   values = as.character(0:16)
  #   # values = c("\u0030", "\u0031", "\u0032", "\u0033", "\u0034", "\u0035", "\u0036", "\u0037", "\u0038", "\u0039", "\u0031\u0030", "\u0031\u0031", "\u0031\u0032", "\u0031\u0033", "\u0031\u0034", "\u0031\u0035")
  #   # values = rejection_values
  #   # values = c("\u25cb", "\u25d6", "\u25d7")
  # ) +
  facet_wrap(~model) +
  theme_bw(base_size = 16) +
  theme(
    legend.position = "top",
    panel.grid = element_blank()
  ) +
  labs(
    x = "Experiment",
    y = "% of time Satisfied"
  )

# exp1_relative %>%
#   filter(swapped==FALSE) %>%
#   inner_join(model_meta) %>%
#   ggplot(aes(params, satisfied, color = class, shape = instruct)) +
#   geom_point() +
#   facet_grid(instruct~exp)


exp1_relative %>%
  group_by(model, swapped, exp) %>%
  summarize(
    n = n(),
    sd = sd(satisfied),
    ste = qt(0.05/2, n-1, lower.tail = FALSE) * sd/sqrt(n),
    avg_satisfied = mean(satisfied),
    max_satisfied = max(satisfied)
  ) %>% 
  ungroup() %>%
  filter(swapped==FALSE) %>%
  pivot_longer(avg_satisfied:max_satisfied, names_to = "metric", values_to = "score") %>%
  mutate(
    metric = str_remove(metric, "_satisfied"),
    ste = case_when(
      metric == "max" ~ 0,
      TRUE ~ ste
    )
  ) %>%
  inner_join(model_meta) %>%
  mutate(
    exp = case_when(
      exp == "exp1" ~ "Wait ARC > No ARC (relative)",
      exp == "exp1_2" ~ "Wait MC > Wait ARC",
      exp == "exp2" ~ "No MC > No ARC"
    ),
    exp = factor(exp, levels = c("Wait ARC > No ARC (relative)", "Wait MC > Wait ARC", "No MC > No ARC")),
  ) %>%
  ggplot(aes(params/1e9, score, color = class, fill = class, shape = instruct, group = interaction(class, instruct))) +
  geom_point(size = 2.5) +
  geom_line() +
  geom_ribbon(aes(ymin = score-ste, ymax = score+ste), color = NA, alpha = 0.3) +
  geom_hline(yintercept = 0.5, linetype = "dashed") +
  facet_grid(metric~exp) +
  # scale_x_log10(limits = c(0.5, 8), breaks = c(0.5,2,4,6,8), labels = scales::label_number_auto()) +
  scale_x_log10(limits = c(0.5, 8), breaks = c(0.5,1,2,4,6,8), labels = c("1/2", "1", "2", "4", "6", "8")) +
  scale_y_continuous(limits = c(0,1), labels = scales::percent_format()) +
  scale_color_brewer(palette = "Dark2", aesthetics = c("color", "fill")) +
  theme_bw(base_size = 16) +
  theme(
    legend.position = "top"
  ) +
  labs(
    x = "Parameters (in Billions)",
    y = "% Satisfied"
  )

exp1_relative %>%
  group_by(model, swapped, exp) %>%
  summarize(
    n = n(),
    sd = sd(satisfied),
    ste = qt(0.05/2, n-1, lower.tail = FALSE) * sd/sqrt(n),
    avg_satisfied = mean(satisfied),
    max_satisfied = max(satisfied)
  ) %>% 
  ungroup() %>%
  filter(swapped==FALSE) %>%
  pivot_longer(avg_satisfied:max_satisfied, names_to = "metric", values_to = "score") %>%
  mutate(
    metric = str_remove(metric, "_satisfied"),
    ste = case_when(
      metric == "max" ~ 0,
      TRUE ~ ste
    )
  ) %>%
  ggplot(aes(exp, score, color = metric, group = metric)) +
  geom_point(size = 2) +
  geom_linerange(aes(ymin = score-ste, ymax = score+ste))+
  geom_hline(yintercept = 0.5, linetype = "dashed") +
  scale_y_continuous(limits = c(0,1)) +
  facet_wrap(~model)

exp1_prefix <- arc_results %>%
  inner_join(kim22_arc) %>%
  select(-name1, -name2) %>%
  mutate(
    exp1_1 = wait_prefix_c1 - no_prefix_c1,
    exp2_1 = no_prefix_c2 - no_prefix_c1,
  ) %>% 
  select(model, idx, item, rejection_id, swapped, exp1_1:exp2_1) %>%
  pivot_longer(exp1_1:exp2_1, names_to = "exp", values_to = "score") %>%
  group_by(model, rejection_id, swapped, exp) %>%
  summarize(
    satisfied = mean(score > 0)
  ) %>%
  ungroup()

exp1_prefix %>%
  ggplot(aes(exp, satisfied, color = swapped, shape = swapped)) +
  geom_jitter(size = 2, width = 0.2, alpha = 0.6) +
  facet_wrap(~model) +
  theme_bw()


exp1_prefix %>%
  group_by(model, swapped, exp) %>%
  summarize(
    n = n(),
    sd = sd(satisfied),
    ste = qt(0.05/2, n-1, lower.tail = FALSE) * sd/sqrt(n),
    avg_satisfied = mean(satisfied),
    max_satisfied = max(satisfied)
  ) %>% 
  ungroup() %>%
  pivot_longer(avg_satisfied:max_satisfied, names_to = "metric", values_to = "score") %>%
  mutate(
    metric = str_remove(metric, "_satisfied"),
    ste = case_when(
      metric == "max" ~ 0,
      TRUE ~ ste
    )
  ) %>%
  ggplot(aes(exp, score, color = metric, group = metric)) +
  geom_point(size = 2) +
  geom_linerange(aes(ymin = score-ste, ymax = score+ste))+
  facet_wrap(~model)

exp1_relative %>% 
  ggplot(aes(exp, satisfied, shape = swapped, group = swapped)) +
  geom_point(size = 2) +
  geom_line() +
  geom_hline(yintercept = 0.5) +
  facet_grid(rejection_id~model) +
  theme_bw() +
  theme(
    legend.position = "none",
    panel.grid = element_blank()
  )

exp1_prefix %>% 
  ggplot(aes(exp, satisfied, shape = swapped, group = swapped)) +
  geom_point(size = 2) +
  geom_line() +
  geom_hline(yintercept = 0.5) +
  facet_grid(rejection_id~model) +
  theme_bw() +
  theme(
    legend.position = "none",
    panel.grid = element_blank()
  )



### COORD

kim22_coord <- read_csv("data/stimuli/kim22-coord.csv")

coord_results <- dir_ls("data/results/kim22-coord-metrics/", regexp = "*.csv") %>%
  map_df(read_csv, .id = "model") %>%
  mutate(
    model = str_extract(model, "(?<=kim22-coord-metrics/)(.*)(?=.csv)"),
    model = str_replace(model, "(meta-llama_|Qwen_)", ""),
    stimuli = "COORD"
  ) %>%
  mutate(
    # no_c1_dcpmi = no_prefix_c1 - no_c1,
    # no_c2_dcpmi = no_prefix_c2 - no_c2,
    # wait_c1_dcpmi = wait_prefix_c1 - wait_c1,
    # wait_c2_dcpmi = wait_prefix_c2 - wait_c2,
    exp1 = wait_prefix_c2 - wait_prefix_c1,
    exp2 = no_prefix_c2 - no_prefix_c1
  ) %>%
  # select(model, idx, no_c1_dcpmi:wait_c2_dcpmi) %>%
  inner_join(kim22_arc)


coord_experiment <- coord_results %>%
  select(model, idx, item, rejection_id, swapped, exp1:exp2) %>%
  pivot_longer(exp1:exp2, names_to = "exp", values_to = "score") %>%
  group_by(model, rejection_id, swapped, exp) %>%
  summarize(
    satisfied = mean(score > 0)
  ) %>%
  ungroup()

coord_experiment %>%
  group_by(model, swapped, exp) %>%
  summarize(
    n = n(),
    sd = sd(satisfied),
    ste = qt(0.05/2, n-1, lower.tail = FALSE) * sd/sqrt(n),
    avg_satisfied = mean(satisfied),
    max_satisfied = max(satisfied)
  ) %>% 
  ungroup() %>%
  filter(swapped==FALSE) %>%
  pivot_longer(avg_satisfied:max_satisfied, names_to = "metric", values_to = "score") %>%
  mutate(
    metric = str_remove(metric, "_satisfied"),
    ste = case_when(
      metric == "max" ~ 0,
      TRUE ~ ste
    )
  ) %>%
  inner_join(model_meta) %>%
  mutate(
    exp = case_when(
      exp == "exp1" ~ "Wait VP1 vs. Wait VP2",
      exp == "exp2" ~ "No VP1 vs. No VP2"
    ),
    exp = factor(exp, levels = c("Wait VP1 vs. Wait VP2", "No VP1 vs. No VP2")),
  ) %>%
  ggplot(aes(params/1e9, score, color = class, fill = class, shape = instruct, group = interaction(class, instruct))) +
  geom_point(size = 2.5) +
  geom_line() +
  geom_ribbon(aes(ymin = score-ste, ymax = score+ste), color = NA, alpha = 0.3) +
  geom_hline(yintercept = 0.5, linetype = "dashed") +
  facet_grid(metric~exp) +
  # scale_x_log10(limits = c(0.5, 8), breaks = c(0.5,2,4,6,8), labels = scales::label_number_auto()) +
  scale_x_log10(limits = c(0.5, 8), breaks = c(0.5,1,2,4,6,8), labels = c("1/2", "1", "2", "4", "6", "8")) +
  scale_y_continuous(limits = c(0,1), labels = scales::percent_format()) +
  scale_color_brewer(palette = "Dark2", aesthetics = c("color", "fill")) +
  theme_bw(base_size = 16) +
  theme(
    legend.position = "top"
  ) +
  labs(
    x = "Parameters (in Billions)",
    y = "% Satisfied"
  )





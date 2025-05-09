library(tidyverse)
library(fs)

kim22_arc <- read_csv("data/stimuli/kim22-arc-dcpmi.csv")

arc_results <- dir_ls("data/results/kim22-arc-relative/", regexp = "*.csv") %>%
  map_df(read_csv, .id = "model") %>%
  mutate(
    model = str_extract(model, "(?<=kim22-arc-relative/)(.*)(?=.csv)"),
    model = str_replace(model, "(meta-llama_|Qwen_)", ""),
    stimuli = "ARC"
  )

arc_results %>%
  mutate(
    # no_c1_dcpmi = no_prefix_c1 - no_c1,
    # no_c2_dcpmi = no_prefix_c2 - no_c2,
    # wait_c1_dcpmi = wait_prefix_c1 - wait_c1,
    # wait_c2_dcpmi = wait_prefix_c2 - wait_c2,
    no_c1_relative = exp(no_prefix_c1)/(exp(no_prefix_c1) + exp(no_control)),
    no_c2_relative = exp(no_prefix_c2)/(exp(no_prefix_c2) + exp(no_control)),
    wait_c1_relative = exp(wait_prefix_c1)/(exp(wait_prefix_c1) + exp(wait_control)),
    wait_c2_relative = exp(wait_prefix_c2)/(exp(wait_prefix_c2) + exp(wait_control)),
  ) %>%
  # select(model, idx, no_c1_dcpmi:wait_c2_dcpmi) %>%
  inner_join(kim22_arc) %>%
  select(-name1, -name2) %>%
  mutate(
    # exp1_1 = wait_c1_dcpmi - no_c1_dcpmi,
    exp1_1 = wait_c1_relative - no_c1_relative,
    # exp1_1 = wait_prefix_c1 - no_prefix_c1,
    # exp1_2 = wait_c2_dcpmi - no_c2_dcpmi,
    # exp1_2 = wait_prefix_c2 - no_prefix_c2,
    # exp1_3 = wait_prefix_c2 - wait_prefix_c1,
    # exp1_3 = wait_c1_dcpmi - wait_c2_dcpmi,
    # exp2_1 = no_c2_dcpmi - no_c1_dcpmi,
    # exp2_1 = no_prefix_c2 - no_prefix_c1,
    exp2_1 = no_c2_relative - no_c1_relative,
    # exp2_2 = (no_c2_dcpmi - no_c1_dcpmi) - (wait_c2_dcpmi - wait_c1_dcpmi)
    # exp2_2 = (no_prefix_c2 - no_prefix_c1) - (wait_prefix_c2 - wait_prefix_c1)
  ) %>%
  select(model, idx, item, rejection_id, swapped, exp1_1:exp2_1) %>%
  pivot_longer(exp1_1:exp2_1, names_to = "exp", values_to = "score") %>%
  group_by(model, rejection_id, swapped, exp) %>%
  summarize(
    ste = 1.96 * plotrix::std.error(score),
    score = mean(score)
  ) %>%
  # filter(swapped == FALSE) %>%
  ggplot(aes(exp, score, shape = swapped, color = factor(rejection_id), group = interaction(swapped, rejection_id))) +
  geom_point(size = 2) +
  geom_line() +
  geom_linerange(aes(ymin = score - ste, ymax = score + ste)) +
  geom_hline(yintercept = 0) +
  facet_grid(rejection_id~model) +
  theme_bw() +
  theme(
    legend.position = "none",
    panel.grid = element_blank()
  )

kim22_arc %>%
  distinct(rejection_id, no, wait)


arc_results %>%
  mutate(
    no_control = no_control * 5,
    wait_control = wait_control * 5,
    no_prefix_c1 = no_prefix_c1 * 4,
    no_prefix_c2 = no_prefix_c2 * 4,
    wait_prefix_c1 = wait_prefix_c1 * 4,
    wait_prefix_c2 = wait_prefix_c2 * 4,
    # no_c1_dcpmi = no_prefix_c1 - no_c1,
    # no_c2_dcpmi = no_prefix_c2 - no_c2,
    # wait_c1_dcpmi = wait_prefix_c1 - wait_c1,
    # wait_c2_dcpmi = wait_prefix_c2 - wait_c2,
    no_c1_relative = exp(no_prefix_c1)/(exp(no_prefix_c1) + exp(no_control)),
    no_c2_relative = exp(no_prefix_c2)/(exp(no_prefix_c2) + exp(no_control)),
    wait_c1_relative = exp(wait_prefix_c1)/(exp(wait_prefix_c1) + exp(wait_control)),
    wait_c2_relative = exp(wait_prefix_c2)/(exp(wait_prefix_c2) + exp(wait_control)),
  ) %>%
  # select(model, idx, no_c1_dcpmi:wait_c2_dcpmi) %>%
  inner_join(kim22_arc) %>%
  select(-name1, -name2) %>%
  mutate(
    # exp1_1 = wait_c1_dcpmi - no_c1_dcpmi,
    # exp1_1 = wait_c1_relative - no_c1_relative,
    exp1_1 = wait_prefix_c1 - no_prefix_c1,
    # exp1_2 = wait_c2_dcpmi - no_c2_dcpmi,
    # exp1_2 = wait_prefix_c2 - no_prefix_c2,
    # exp1_3 = wait_prefix_c2 - wait_prefix_c1,
    # exp1_3 = wait_c1_dcpmi - wait_c2_dcpmi,
    # exp2_1 = no_c2_dcpmi - no_c1_dcpmi,
    exp2_1 = no_prefix_c2 - no_prefix_c1,
    # exp2_1 = no_c2_relative - no_c1_relative,
    # exp2_2 = (no_c2_dcpmi - no_c1_dcpmi) - (wait_c2_dcpmi - wait_c1_dcpmi)
    # exp2_2 = (no_prefix_c2 - no_prefix_c1) - (wait_prefix_c2 - wait_prefix_c1)
  ) %>%
  select(model, idx, item, rejection_id, swapped, exp1_1:exp2_1) %>%
    pivot_longer(exp1_1:exp2_1, names_to = "exp", values_to = "score") %>%
    group_by(model, rejection_id, swapped, exp) %>%
    summarize(
      satisfied = mean(score > 0)
    ) %>% 
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

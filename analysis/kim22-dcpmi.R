library(tidyverse)
library(fs)

kim22_arc <- read_csv("data/stimuli/kim22-arc-dcpmi.csv")

arc_results <- dir_ls("data/results/kim22-arc-dcpmi/", regexp = "*.csv") %>%
  map_df(read_csv, .id = "model") %>%
  mutate(
    model = str_extract(model, "(?<=kim22-arc-dcpmi/)(.*)(?=.csv)"),
    model = str_replace(model, "(meta-llama_|Qwen_)", ""),
    stimuli = "ARC"
  )

arc_results %>% count(model)
# exp1_1 = waitno_arc - no_arc, # should be large
# exp1_2 = waitno_mc - no_mc, # should be close to 0
# exp2_1 = no_mc - no_arc, # should be large
# exp2_2 = (no_mc - no_arc) - (waitno_mc - waitno_arc) # should be large

arc_results %>%
  mutate(
    no_c1_dcpmi = no_prefix_c1 - no_c1,
    no_c2_dcpmi = no_prefix_c2 - no_c2,
    wait_c1_dcpmi = wait_prefix_c1 - wait_c1,
    wait_c2_dcpmi = wait_prefix_c2 - wait_c2,
  ) %>%
  # select(model, idx, no_c1_dcpmi:wait_c2_dcpmi) %>%
  inner_join(kim22_arc) %>%
  select(-name1, -name2) %>%
  mutate(
    exp1_1 = wait_c1_dcpmi - no_c1_dcpmi,
    # exp1_1 = wait_prefix_c1 - no_prefix_c1,
    exp1_2 = wait_c2_dcpmi - no_c2_dcpmi,
    # exp1_2 = wait_prefix_c2 - no_prefix_c2,
    exp1_3 = wait_prefix_c2 - wait_prefix_c1,
    # exp1_3 = wait_c1_dcpmi - wait_c2_dcpmi,
    # exp2_1 = no_c2_dcpmi - no_c1_dcpmi,
    exp2_1 = no_prefix_c2 - no_prefix_c1,
    # exp2_2 = (no_c2_dcpmi - no_c1_dcpmi) - (wait_c2_dcpmi - wait_c1_dcpmi)
    exp2_2 = (no_prefix_c2 - no_prefix_c1) - (wait_prefix_c2 - wait_prefix_c1)
  ) %>%
  select(model, idx, item, rejection_id, swapped, exp1_1:exp2_2) %>%
  pivot_longer(exp1_1:exp2_2, names_to = "exp", values_to = "score") %>%
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
  # pivot_longer(no_c1_dcpmi:wait_c2_dcpmi, names_to = "measure", values_to = "score") %>%
  # # count(model, item, swapped, measure)
  # group_by(model, item, swapped, measure) %>%
  # slice_max(score, with_ties = FALSE, n = 1) %>% 
  # ungroup() %>% View()


arc_results %>%
  mutate(
    no_c1_dcpmi = no_prefix_c1 - no_c1,
    no_c2_dcpmi = no_prefix_c2 - no_c2,
    wait_c1_dcpmi = wait_prefix_c1 - wait_c1,
    wait_c2_dcpmi = wait_prefix_c2 - wait_c2,
  ) %>%
  # select(model, idx, no_c1_dcpmi:wait_c2_dcpmi) %>%
  inner_join(kim22_arc) %>%
  select(-name1, -name2) %>%
  mutate(
    exp1_1 = wait_c1_dcpmi - no_c1_dcpmi,
    # exp1_1 = wait_prefix_c1 - no_prefix_c1,
    exp1_2 = wait_c2_dcpmi - no_c2_dcpmi,
    # exp1_2 = no_c2_dcpmi - wait_c2_dcpmi,
    # exp1_2 = wait_prefix_c2 - no_prefix_c2,
    # exp2_1 = no_c2_dcpmi - no_c1_dcpmi,
    exp1_3 = wait_prefix_c2 - wait_prefix_c1,
    # exp1_3 = wait_c1_dcpmi - wait_c2_dcpmi,
    exp2_1 = no_prefix_c2 - no_prefix_c1,
    # exp2_2 = (no_c2_dcpmi - no_c1_dcpmi) - (wait_c2_dcpmi - wait_c1_dcpmi)
    exp2_2 = (no_prefix_c2 - no_prefix_c1) - (wait_prefix_c2 - wait_prefix_c1)
  ) %>%
  select(model, idx, item, rejection_id, swapped, exp1_1:exp2_2) %>%
  pivot_longer(exp1_1:exp2_2, names_to = "exp", values_to = "score") %>%
  group_by(model, rejection_id, swapped, exp) %>%
  summarize(
    satisfied = mean(score > 0)
  ) %>% 
  ggplot(aes(exp, satisfied, shape = swapped)) +
  geom_point(size = 2) +
  geom_hline(yintercept = 0.5) +
  facet_grid(rejection_id~model) +
  theme_bw() +
  theme(
    legend.position = "none",
    panel.grid = element_blank()
  )

library(tidyverse)

kim22 <- read_csv("data/stimuli/kim22-full.csv")

results_raw <- fs::dir_ls("data/results/kim22/", regexp="*.csv") %>%
  map_df(read_csv, .id = "file") %>%
  mutate(
    model = str_remove(file, "data/results/kim22/"),
    model = str_remove(model, "\\.csv"),
    model = str_remove(model, "(EleutherAI_|allenai_|distilbert_|facebook_|google_)"),
    model = str_remove(model, "-deduped")
  ) %>%
  filter(!str_detect(model, "OLMo")) %>%
  select(-file)

# accuracy
results_raw %>%
  pivot_wider(names_from = type, values_from = logprob) %>%
  group_by(model) %>%
  summarize(
    exp1 = mean(neg_arc_good > neg_arc_bad), # "Wait no for ARC > no for ARC"
    exp2 = mean(neg_main_good > neg_arc_bad) # "No for MC > No for ARC"
  )

# t-test for exp 1

results_raw %>%
  pivot_wider(names_from = type, values_from = logprob) %>%
  group_by(model) %>%
  nest() %>%
  mutate(
    t_test = map(data, function(x) {
      t.test(x$neg_arc_good, x$neg_arc_bad) %>% broom::tidy()
    })
  ) %>%
  unnest(t_test)

# exp 2 t-test
results_raw %>%
  pivot_wider(names_from = type, values_from = logprob) %>%
  group_by(model) %>%
  nest() %>%
  mutate(
    t_test = map(data, function(x) {
      t.test(x$neg_main_good, x$neg_arc_bad) %>% broom::tidy()
    })
  ) %>%
  unnest(t_test)



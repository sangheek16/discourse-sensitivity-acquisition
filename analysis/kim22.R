library(tidyverse)

kim22 <- read_csv("data/stimuli/kim22-full.csv") %>%
  rename(old_type = type) %>%
  mutate(
    header = case_when(
      str_detect(prefix, "Wait no,") ~ "waitno",
      TRUE ~ "no"
    ),
    target = case_when(
      str_detect(old_type, "main") ~ "mc",
      TRUE ~ "arc"
    )
  ) %>%
  unite(type, header:target, remove=FALSE)

results_raw <- fs::dir_ls("data/results/kim22/", regexp="*.csv") %>%
  map_df(read_csv, .id = "file") %>%
  mutate(
    model = str_remove(file, "data/results/kim22/"),
    model = str_remove(model, "\\.csv"),
    model = str_remove(model, "(EleutherAI_|allenai_|distilbert_|facebook_|google_)"),
    model = str_remove(model, "-deduped")
  ) %>%
  # filter(!str_detect(model, "OLMo")) %>%
  select(-file) %>%
  rename(old_type = type) %>%
  inner_join(kim22 %>% select(idx, item, old_type, type))

# accuracy
kim22_exps <- results_raw %>%
  select(-old_type) %>%
  pivot_wider(names_from = type, values_from = logprob) %>%
  mutate(
    exp1_1 = waitno_arc - no_arc, # should be large
    exp1_2 = waitno_mc - no_mc, # should be close to 0
    exp2_1 = no_mc - no_arc, # should be large
    exp2_2 = (no_mc - no_arc) - (waitno_mc - waitno_arc) # should be large
  ) %>%
  select(idx, item, model, exp1_1, exp1_2, exp2_1, exp2_2)


# p(something | sentence. wait no)/p(something | wait no) vs. p(something | sentence. no)/p(something | no)

kim22_exps %>%
  pivot_longer(exp1_1:exp2_2, names_to = "exp", values_to = "score") %>%
  group_by(model, exp) %>%
  summarize(
    ste = 1.96 * plotrix::std.error(score),
    score = mean(score)
  ) %>%
  ggplot(aes(exp, score, color = exp)) +
  geom_point(size = 2) +
  geom_linerange(aes(ymin = score-ste, ymax = score+ste)) + 
  geom_hline(yintercept = 0.0, linetype="dashed") +
  facet_wrap(~model)

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



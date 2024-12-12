library(tidymodels)
library(tidyverse)
library(vroom)
library(embed)
library(discrim)
library(doParallel)
library(rules)
library(themis)
library(C50)

num_cores <- detectCores()
cl <- makePSOCKcluster(num_cores)
registerDoParallel(cl)

# Read in data
train_data <- vroom("./train.csv")
test_data <- vroom("./test.csv")
sample_sub <- vroom("./sampleSubmission.csv")

# Make ACTION a factor
train_data$ACTION <- as.factor(train_data$ACTION)

# Create Recipe
amazon_recipe <- recipe(ACTION ~ ., data = train_data) %>%
  step_mutate_at(all_predictors(), fn = factor) %>% 
  step_lencode_mixed(all_nominal_predictors(), outcome = vars(ACTION)) %>% 
  step_normalize(all_numeric_predictors()) %>% 
  step_rm(ROLE_CODE) %>% 
  step_smote(all_outcomes(), neighbors = 5)
prepped_amazon_recipe <- prep(amazon_recipe)
bake(prepped_amazon_recipe, new_data = train_data)

# C5_rules Model
c5_model <- C5_rules(trees = 500,
                     min_n = tune()) %>% 
  set_engine("C5.0") %>% 
  set_mode("classification") %>% 
  translate()

# Set workflow
c5_wf <- workflow() %>% 
  add_recipe(amazon_recipe) %>% 
  add_model(c5_model)

# Set tuning grid
tuning_grid <- grid_regular(min_n(),
                            levels = 5)

# Set number of folds
folds <- vfold_cv(train_data, v = 5, repeats = 1)

# CV
CV_results <- c5_wf %>% 
  tune_grid(resamples = folds,
            grid = tuning_grid,
            metrics = metric_set(roc_auc))

# Select best tuning parameter
best_tune <- CV_results %>% 
  select_best(metric = "roc_auc")
best_tune

# Finalize workflow
c5_final_wf <- c5_wf %>% 
  finalize_workflow(best_tune) %>% 
  fit(data = train_data)

# Generate predictions
c5_preds <- predict(c5_final_wf,
                    new_data = test_data,
                    type = "prob")


# Prepare predictions for Kaggle submissions
kaggle_submission <- c5_preds %>% 
  rename(Action = .pred_1) %>% 
  mutate(Id = seq.int(nrow(c5_preds))) %>% 
  select(Id, Action) %>% 
  arrange(Id)

# Write the submission to a csv file
vroom_write(x = kaggle_submission,
            file = "./smote_c5_preds.csv", 
            delim = ",")

stopCluster(cl)

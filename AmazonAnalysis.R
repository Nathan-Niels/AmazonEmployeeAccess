library(tidyverse)
library(tidymodels)
library(embed)
library(vroom)
library(skimr)
library(DataExplorer)
library(ggmosaic)
library(ggplot2)
library(lme4)

# Read in data
train_data <- vroom("C:/Users/nsnie/OneDrive/BYU Classes/Fall 2024/STAT 348/AmazonEmployeeAccess/train.csv")
test_data <- vroom("C:/Users/nsnie/OneDrive/BYU Classes/Fall 2024/STAT 348/AmazonEmployeeAccess/test.csv")
sample_sub <- vroom("C:/Users/nsnie/OneDrive/BYU Classes/Fall 2024/STAT 348/AmazonEmployeeAccess/sampleSubmission.csv")

glimpse(train_data)
skim(train_data)
plot_bar(train_data)
plot_correlation(train_data)

# Boxplot of ROLE_TITLE
title_plot <- ggplot(data = train_data) +
  geom_boxplot(aes(x = ACTION, y = ROLE_TITLE))
title_plot

# Boxplot of ROLE_FAMILY
fam_plot <- ggplot(data = train_data) +
  geom_boxplot(aes(x = ACTION, y = ROLE_FAMILY))
fam_plot

# Make ACTION a factor
train_data$ACTION <- as.factor(train_data$ACTION)

# Recipe
amazon_recipe <- recipe(ACTION ~ ., data = train_data) %>%
  step_mutate_at(all_predictors(), fn = factor) %>% 
  # step_other(all_predictors(), threshold = 0.04) %>% 
  step_lencode_mixed(all_nominal_predictors(), outcome = vars(ACTION)) %>% 
  step_normalize(all_numeric_predictors()) %>% 
  step_rm(ROLE_CODE)
prepped_amazon_recipe <- prep(amazon_recipe)
bake(prepped_amazon_recipe, new_data = train_data)

view(baked_recipe)

## Logistic Regression
# Logistic regression model
log_reg_model <- logistic_reg() %>% 
  set_engine("glm")

# Logistic regression workflow
log_reg_wf <- workflow() %>% 
  add_recipe(amazon_recipe) %>% 
  add_model(log_reg_model) %>% 
  fit(data = train_data)

# Generate predictions
log_amazon_preds <- predict(log_reg_wf,
                            new_data = test_data,
                            type = "prob")

log_amazon_preds

# Prepare predictions for Kaggle submissions
kaggle_submission <- log_amazon_preds %>% 
  rename(Action = .pred_1) %>% 
  mutate(Id = seq.int(nrow(log_amazon_preds))) %>% 
  select(Id, Action) %>% 
  arrange(Id)

# Write the submission to a csv file
vroom_write(x = kaggle_submission,
            file = "C:/Users/nsnie/OneDrive/BYU Classes/Fall 2024/STAT 348/AmazonEmployeeAccess/log_reg_preds.csv", 
            delim = ",")


## Penalized Logistic Regression
# Create model
plog_reg_model <- logistic_reg(mixture = tune(),
                               penalty = tune()) %>% 
  set_engine("glmnet")

# Create workflow
plog_reg_wf <- workflow() %>% 
  add_recipe(amazon_recipe) %>% 
  add_model(plog_reg_model)

# Create grid of values to tune over
tuning_grid <- grid_regular(penalty(),
                            mixture(),
                            levels = 5)

# Split data into folds
folds <- vfold_cv(train_data, v = 6, repeats = 1)

# Run CV
CV_results <- plog_reg_wf %>% 
  tune_grid(resamples = folds,
            grid = tuning_grid,
            metrics = metric_set(roc_auc))
                                 #f_meas,
                                 #sens,
                                 #recall,
                                 #precision,
                                 #accuracy))

# Find best tuning parameters
best_tune <- CV_results %>% 
  select_best(metric = "roc_auc")
best_tune

# Finalize workflow and fit it
plog_reg_final_wf <- plog_reg_wf %>% 
  finalize_workflow(best_tune) %>% 
  fit(data = train_data)

# Generate predictions
plog_reg_preds <- predict(plog_reg_final_wf,
                          new_data = test_data,
                          type = "prob")

# Prepare predictions for Kaggle submissions
kaggle_submission <- plog_reg_preds %>% 
  rename(Action = .pred_1) %>% 
  mutate(Id = seq.int(nrow(plog_reg_preds))) %>% 
  select(Id, Action) %>% 
  arrange(Id)

# Write the submission to a csv file
vroom_write(x = kaggle_submission,
            file = "C:/Users/nsnie/OneDrive/BYU Classes/Fall 2024/STAT 348/AmazonEmployeeAccess/plog_reg_preds.csv", 
            delim = ",")

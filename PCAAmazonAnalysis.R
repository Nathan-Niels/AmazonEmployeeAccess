library(tidymodels)
library(tidyverse)
library(vroom)
library(embed)
library(discrim)
library(naivebayes)
library(lme4)
library(knn)
library(doParallel)

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
  step_pca(all_predictors(), threshold = 0.85)
prepped_amazon_recipe <- prep(amazon_recipe)
bake(prepped_amazon_recipe, new_data = train_data)

# NB Model
nb_model <- naive_Bayes(Laplace = tune(),
                        smoothness = tune()) %>% 
  set_mode("classification") %>% 
  set_engine("naivebayes")

# Set workflow
nb_wf <- workflow() %>% 
  add_recipe(amazon_recipe) %>% 
  add_model(nb_model)

# Set tuning grid
tuning_grid <- grid_regular(Laplace(),
                            smoothness(),
                            levels = 5)

# Set number of folds
folds <- vfold_cv(train_data, v = 5, repeats = 1)

# CV
CV_results <- nb_wf %>% 
  tune_grid(resamples = folds,
            grid = tuning_grid,
            metrics = metric_set(roc_auc))

# Select best tuning parameter
best_tune <- CV_results %>% 
  select_best(metric = "roc_auc")
best_tune

# Finalize workflow
nb_final_wf <- nb_wf %>% 
  finalize_workflow(best_tune) %>% 
  fit(data = train_data)

# Generate predictions
nb_preds <- predict(nb_final_wf,
                     new_data = test_data,
                     type = "prob")

# Prepare predictions for Kaggle submissions
kaggle_submission <- nb_preds %>% 
  rename(Action = .pred_1) %>% 
  mutate(Id = seq.int(nrow(nb_preds))) %>% 
  select(Id, Action) %>% 
  arrange(Id)

# Write the submission to a csv file
vroom_write(x = kaggle_submission,
            file = "./pca_nb_preds.csv", 
            delim = ",")

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
            file = "./pca_log_reg_preds.csv", 
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
            file = "./pca_plog_reg_preds.csv", 
            delim = ",")

# KNN Model
knn_model <- nearest_neighbor(neighbors = tune()) %>% 
  set_mode("classification") %>% 
  set_engine("kknn")

# Set workflow
knn_wf <- workflow() %>% 
  add_recipe(amazon_recipe) %>% 
  add_model(knn_model)

# Set tuning grid
tuning_grid <- grid_regular(neighbors(),
                            levels = 5)

# Set number of folds
folds <- vfold_cv(train_data, v = 5, repeats = 1)

# CV
CV_results <- knn_wf %>% 
  tune_grid(resamples = folds,
            grid = tuning_grid,
            metrics = metric_set(roc_auc))

# Select best tuning parameter
best_tune <- CV_results %>% 
  select_best(metric = "roc_auc")
best_tune

# Finalize workflow
knn_final_wf <- knn_wf %>% 
  finalize_workflow(best_tune) %>% 
  fit(data = train_data)

# Generate predictions
knn_preds <- predict(knn_final_wf,
                     new_data = test_data,
                     type = "prob")

# Prepare predictions for Kaggle submissions
kaggle_submission <- knn_preds %>% 
  rename(Action = .pred_1) %>% 
  mutate(Id = seq.int(nrow(knn_preds))) %>% 
  select(Id, Action) %>% 
  arrange(Id)

# Write the submission to a csv file
vroom_write(x = kaggle_submission,
            file = "./pca_knn_preds.csv", 
            delim = ",")

# RF Model
rf_model <- rand_forest(mtry = tune(),
                        min_n = tune(),
                        trees = 500) %>% 
  set_mode("classification") %>% 
  set_engine("ranger")

# Set workflow
rf_wf <- workflow() %>% 
  add_recipe(amazon_recipe) %>% 
  add_model(rf_model)

# Finalize mtry() parameter
final_par <- extract_parameter_set_dials(rf_model) %>% 
  finalize(train_data)

# Set tuning grid
tuning_grid <- grid_regular(final_par,
                            levels = 5)

# Set number of folds
folds <- vfold_cv(train_data, v = 5, repeats = 1)

# CV
CV_results <- rf_wf %>% 
  tune_grid(resamples = folds,
            grid = tuning_grid,
            metrics = metric_set(roc_auc))

# Select best tuning parameter
best_tune <- CV_results %>% 
  select_best(metric = "roc_auc")
best_tune

# Finalize workflow
rf_final_wf <- rf_wf %>% 
  finalize_workflow(best_tune) %>% 
  fit(data = train_data)

# Generate predictions
rf_preds <- predict(rf_final_wf,
                    new_data = test_data,
                    type = "prob")

# Prepare predictions for Kaggle submissions
kaggle_submission <- rf_preds %>% 
  rename(Action = .pred_1) %>% 
  mutate(Id = seq.int(nrow(rf_preds))) %>% 
  select(Id, Action) %>% 
  arrange(Id)

# Write the submission to a csv file
vroom_write(x = kaggle_submission,
            file = "./pca_rf_preds.csv", 
            delim = ",")

stopCluster(cl)

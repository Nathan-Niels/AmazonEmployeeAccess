library(tidymodels)
library(tidyverse)
library(vroom)
library(embed)
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

# SVMS Radial Model
radial_mod <- svm_rbf(rbf_sigma = tune(),
                      cost = tune()) %>% 
  set_mode("classification") %>% 
  set_engine("kernlab")

# Set workflow
radial_wf <- workflow() %>% 
  add_recipe(amazon_recipe) %>% 
  add_model(radial_mod)

# Set tuning grid
tuning_grid <- grid_regular(rbf_sigma(),
                            cost(),
                            levels = 5)

# Set number of folds
folds <- vfold_cv(train_data, v = 5, repeats = 1)

# CV
CV_results <- radial_wf %>% 
  tune_grid(resamples = folds,
            grid = tuning_grid,
            metrics = metric_set(roc_auc))

# Select best tuning parameter
best_tune <- CV_results %>% 
  select_best(metric = "roc_auc")
best_tune

# Finalize workflow
radial_final_wf <- radial_wf %>% 
  finalize_workflow(best_tune) %>% 
  fit(data = train_data)

# Generate predictions
radial_preds <- predict(radial_final_wf,
                     new_data = test_data,
                     type = "prob")

# Prepare predictions for Kaggle submissions
kaggle_submission <- radial_preds %>% 
  rename(Action = .pred_1) %>% 
  mutate(Id = seq.int(nrow(radial_preds))) %>% 
  select(Id, Action) %>% 
  arrange(Id)

# Write the submission to a csv file
vroom_write(x = kaggle_submission,
            file = "./smote_svms_rad_preds.csv", 
            delim = ",")

stopCluster(cl)

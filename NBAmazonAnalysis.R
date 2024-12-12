library(tidymodels)
library(tidyverse)
library(vroom)
library(embed)
library(discrim)
library(naivebayes)

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
  step_rm(ROLE_CODE)
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
            file = "./nb_preds.csv", 
            delim = ",")

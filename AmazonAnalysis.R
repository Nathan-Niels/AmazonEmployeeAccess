library(tidyverse)
library(tidymodels)
library(embed)
library(vroom)
library(skimr)
library(DataExplorer)
library(ggmosaic)
library(ggplot2)

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
  step_other(all_predictors(), threshold = 0.04) %>% 
  step_dummy(all_nominal_predictors())
prepped_amazon_recipe <- prep(amazon_recipe)
bake(prepped_amazon_recipe, new_data = train_data)


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
  rename(Action = .pred_0) %>% 
  mutate(Id = seq.int(nrow(log_amazon_preds))) %>% 
  select(Id, Action) %>% 
  arrange(Id)

# Write the submission to a csv file
vroom_write(x = kaggle_submission,
            file = "C:/Users/nsnie/OneDrive/BYU Classes/Fall 2024/STAT 348/AmazonEmployeeAccess/log_reg_preds.csv", 
            delim = ",")

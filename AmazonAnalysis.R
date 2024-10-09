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

glimpse(train_data)
skim(train_data)
plot_bar(train_data)
plot_correlation(train_data)

# Boxplot of Family
fam_plot <- ggplot(data = train_data) +
  geom_boxplot(aes(x = ACTION, y = ROLE_FAMILY))
fam_plot

amazon_recipe <- recipe(ACTION ~ ., data = train_data) %>% 
  step_mutate_at(all_predictors(), fn = factor) %>% 
  step_other(all_predictors(), threshold = 0.001) %>% 
  step_dummy(all_nominal_predictors())
prepped_amazon_recipe <- prep(amazon_recipe)
baked_amazon_recipe <- bake(prepped_amazon_recipe, new_data = train_data)

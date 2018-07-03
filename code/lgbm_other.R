# This is a modified version of the public kernel by Pranav Pandya:
# https://www.kaggle.com/pranav84/single-lightgbm-in-r-with-75-mln-rows-lb-0-9690
# Many thanks to Pranav for his work on the original code.
#
# What is new?
# 1. Added 4 new Features.
#  For simplicity define an (ip,device,os) combination as a user instance.
#  The 4 new features are,
#  1) total count of a unique user instance.
#  2) a running count of this particular user instance's visit number.
#     For the first time a user instance appears, its 1 , the next time its 2 and so on.
#  3) total count of a unique user instance and a unique app.
#  4) a running count of this particular user instance using a particular app.
#     For the first time a user appears with a unique app, its 1 , the next time the same user
#     appearing with the same app its 2 and so on.

# 2. Some cosmetic changes.
#    1) Modified to train on 40 million rows, since we added new features.
#    2) set seed to 1
#    3) validation ratio to 0.9

#######################################################
#IMPORTANT: If you would like to run the code be sure 
#to set the `testing' parameter to FALSE. Otherwise, it
#will only process the first 100000 rows from the training
#data set. I made this parameter to be able to quickly debug
#the code.
#######################################################

if (!require("pacman")) install.packages("pacman")
pacman::p_load(knitr, tidyverse, data.table, lubridate, zoo, DescTools, lightgbm)
set.seed(1)               
options(scipen = 9999, warn = -1, digits= 4)

#######################################################
# Some frequently used control parameters:
####################################################### 
testing = TRUE 
#######################################################
testing_size <- 100000
#######################################################
val_ratio <- 0.90
#######################################################

train_path <- "/home/rjs/바탕화면/adtrack/data/train.csv"
test_path  <- "/home/rjs/바탕화면/adtrack/data/test.csv"

#######################################################

train_col_names <- c("ip", "app", "device", "os", "channel", 
                     "click_time", "attributed_time", "is_attributed")

#######################################################

most_freq_hours_in_test_data <- c("4","5","9","10","13","14")
least_freq_hours_in_test_data <- c("6","11","15")

#######################################################

total_rows <- 184903890

#######################################################

if (!testing){train_rows <- 40000000
skip_rows_train <- total_rows - train_rows
test_rows <- -1L
} else {
  train_rows <- testing_size
  skip_rows_train <- 0
  test_rows <- testing_size
}

#######################################################

#*****************************************************************
# A function for processing the train/test data

#######################################################

process <- function(df) {
  cat("Building new features...\n")
  df <- df %>% mutate(wday = Weekday(click_time), 
                      hour = hour(click_time),
                      in_test_hh = ifelse(hour %in% most_freq_hours_in_test_data, 1,
                                          ifelse(hour %in% least_freq_hours_in_test_data, 3, 2))) %>%
    select(-c(click_time)) %>%
    add_count(ip, wday, in_test_hh) %>% rename("nip_day_test_hh" = n) %>%
    select(-c(in_test_hh)) %>%
    add_count(ip, wday, hour) %>% rename("n_ip" = n) %>%
    add_count(ip, wday, hour, os) %>% rename("n_ip_os" = n) %>% 
    add_count(ip, wday, hour, app) %>% rename("n_ip_app" = n) %>%
    add_count(ip, wday, hour, app, os) %>% rename("n_ip_app_os" = n) %>% 
    add_count(app, wday, hour) %>% rename("n_app" = n) %>%
    select(-c(wday)) %>% select(-c(ip)) 
  return(df)
}

#######################################################

#*****************************************************************
# Preparing the training data

#######################################################

cat("Reading the training data...\n")
train <- fread(train_path, skip = skip_rows_train, nrows = train_rows, colClasses = list(numeric=1:5),
               showProgress = FALSE, col.names = train_col_names) %>% select(-c(attributed_time))           
invisible(gc())

#######################################################

cat("Processing the training data...\n")

train[, UsrappCount:=.N, by=list(ip,app,device,os)]
train[, UsrappNewness:=1:.N, by=list(ip,app,device,os)]
train[, UsrCount:=.N, by=list(ip,device,os)]
train[, UsrNewness:=1:.N, by=list(ip,device,os)]

train <- process(train)
invisible(gc())

#######################################################

cat("The training set has", nrow(train), "rows and", ncol(train), "columns.\n")
cat("The column names of the train are: \n")
cat(colnames(train), "\n")
print("The size of the train is: ") 
print(object.size(train), units = "auto")

#######################################################

#*****************************************************************
# Modelling

#######################################################

print("The table of class unbalance")
table(train$is_attributed)

#######################################################

print("Prepare data for modeling")
library(caret)
train.index <- createDataPartition(train$is_attributed, p = val_ratio, list = FALSE)

#######################################################

dtrain <- train[ train.index,]
valid  <- train[-train.index,]

#######################################################

rm(train)
invisible(gc())

#######################################################

cat("train size : ", dim(dtrain), "\n")
cat("valid size : ", dim(valid), "\n")

#######################################################

categorical_features = c("app", "device", "os", "channel", "hour")

#######################################################

cat("Creating the 'dtrain' for modeling...")
dtrain = lgb.Dataset(data = as.matrix(dtrain[, colnames(dtrain) != "is_attributed"]), 
                     label = dtrain$is_attributed, categorical_feature = categorical_features)

#######################################################

cat("Creating the 'dvalid' for modeling...")
dvalid = lgb.Dataset(data = as.matrix(valid[, colnames(valid) != "is_attributed"]), 
                     label = valid$is_attributed, categorical_feature = categorical_features)

#######################################################

rm(valid)
invisible(gc())

#######################################################

print("Modelling")

params = list(objective = "binary", 
              metric = "auc", 
              learning_rate= 0.1,
              num_leaves= 7,
              max_depth= 4,
              min_child_samples= 100,
              max_bin= 100,
              subsample= 0.7, 
              subsample_freq= 1,
              colsample_bytree= 0.7,
              min_child_weight= 0,
              min_split_gain= 0,
              scale_pos_weight= 99.7)
#######################################################

model <- lgb.train(params, dtrain, valids = list(validation = dvalid), nthread = 4,
                   nrounds = 1500, verbose= 1, early_stopping_rounds = 50, eval_freq = 25)

#######################################################

rm(dtrain, dvalid)
invisible(gc())

#######################################################

cat("Validation AUC @ best iter: ", max(unlist(model$record_evals[["validation"]][["auc"]][["eval"]])), "\n\n")

#######################################################

#*****************************************************************
# Preparing the test data

#######################################################

cat("Reading the test data: ", test_rows, " rows. \n")
test <- fread(test_path, nrows=test_rows, colClasses=list(numeric=2:6), showProgress = FALSE) 

#######################################################

cat("Setting up the submission file... \n")
sub <- data.table(click_id = test$click_id, is_attributed = NA) 
test$click_id <- NULL
invisible(gc())

#######################################################

cat("Processing the test data...\n")
test[, UsrappCount:=.N, by=list(ip,app,device,os)]
test[, UsrappNewness:=1:.N, by=list(ip,app,device,os)]
test[, UsrCount:=.N, by=list(ip,device,os)]
test[, UsrNewness:=1:.N, by=list(ip,device,os)]


test <-process(test)
invisible(gc())

#######################################################

cat("The test set has", nrow(test), "rows and", ncol(test), "columns.\n")
cat("The column names of the test set are: \n")
cat(colnames(test), "\n")
print("The size of the test set is: ") 
print(object.size(test), units = "auto")

#######################################################

#*****************************************************************
# Predictions

#######################################################

cat("Predictions: \n")
preds <- predict(model, data = as.matrix(test[, colnames(test)], n = model$best_iter))

#######################################################

cat("Converting to data frame: \n")
preds <- as.data.frame(preds)

#######################################################

cat("Creating the submission data: \n")
sub$is_attributed = preds

#######################################################

cat("Removing test... \n")
rm(test)
invisible(gc())

#######################################################

cat("Rounding: \n")
sub$is_attributed = round(sub$is_attributed, 4)

#######################################################

cat("Writing into a csv file: \n")
fwrite(sub, "lgb_Usrnewness.csv")

#######################################################

cat("A quick peek at the submission data: \n") 
head(sub, 10)

#######################################################

#*****************************************************************
# Feature importance

#######################################################

cat("Feature importance: ")
kable(lgb.importance(model, percentage = TRUE))

#######################################################

cat("\nAll done!..")

#######################################################
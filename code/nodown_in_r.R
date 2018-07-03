
library(data.table)
library(xgboost)
library(lightgbm)
library(caret)
library(lubridate)
library(dplyr)
library(h2o)
library(fst)

# Nos quedamos solo con las primeras 80000000 lineas de train.csv para no agotar la memoria
# Descartamos las columnas "ip" (1), "click time" (6) y "attributed time" (7)
trainData<-fread('/home/rjs/바탕화면/adtrack/data/train.csv',drop = c(7))
trainData$click_time<- ymd_hms(trainData$click_time)
trainData$click_time <- trainData$click_time + dhours(8)
trainData$hour <- lubridate::hour(trainData$click_time)
trainData$day <- lubridate::day(trainData$click_time)
gc()

testsupData <- fread('/home/rjs/바탕화면/adtrack/data/test_supplement.csv')
testsupData <- testsupData %>% select(-click_id)
testsupData <- testsupData[!duplicated(testsupData),]
testsupData$is_attributed <- 0
testsupData$click_time<-ymd_hms(testsupData$click_time)
testsupData$click_time <- testsupData$click_time + dhours(8)
testsupData$hour <- lubridate::hour(testsupData$click_time)
testsupData$day <- lubridate::day(testsupData$click_time)
gc()

# Descartamos las columnas "ip" (2) y "click time" (7)
testData<-fread('/home/rjs/바탕화면/adtrack/data/test.csv')
testData$click_time<-ymd_hms(testData$click_time)
testData$click_time <- testData$click_time + dhours(8)
gc()

endOfTrainData<-dim(trainData)[1]

totalData <- rbind(trainData, testsupData)
rm(trainData)
gc()

# https://www.kaggle.com/c/talkingdata-adtracking-fraud-detection/discussion/56065

totalData[, UsrappCount:=.N,   by=list(ip, app, device, os)]
totalData[, UsrappRank:=1:.N,  by=list(ip, app, device, os)]
totalData[, UsrCount:=.N,      by=list(ip, os, device)]
totalData[, UsrRank:=1:.N,     by=list(ip, os, device)]
totalData[, nipApp:=.N,        by=list(ip, day, hour, app)]
totalData[, nipAppOs:=.N,      by=list(ip, day, hour, app, os)]
totalData[, napp:=.N,          by=list(app, day, hour)]
totalData[, nappDev:=.N,       by=list(app, day, device)]
totalData[, nappDevOs:=.N,     by=list(app, day, device, os)]
gc()

totalData[, click_time_num := as.numeric(click_time)]

totalData[, by_ipappdeviceoschannel_nextclick1 := c(click_time_num[-1], NA), by = c("ip", "app", "device", "os", "channel")]
totalData[, by_ipappdeviceoschannel_nextclick1 := by_ipappdeviceoschannel_nextclick1 - click_time_num]
totalData[is.na(by_ipappdeviceoschannel_nextclick1), by_ipappdeviceoschannel_nextclick1 := 0]
gc()

totalData[, by_iposdevice_nextclick1 := c(click_time_num[-1], NA), by = c("ip", "os", "device")]
totalData[, by_iposdevice_nextclick1 := by_iposdevice_nextclick1 - click_time_num]
totalData[is.na(by_iposdevice_nextclick1), by_iposdevice_nextclick1 := 0]
gc()

totalData[, by_iposdeviceapp_nextclick1 := c(click_time_num[-1], NA), by = c("ip", "os", "device", "app")]
totalData[, by_iposdeviceapp_nextclick1 := by_iposdeviceapp_nextclick1 - click_time_num]
totalData[is.na(by_iposdeviceapp_nextclick1), by_iposdeviceapp_nextclick1 := 0]
gc()

totalData[, by_ipchannel_prevclick1 := c(NA, click_time_num[-.N]), by = c("ip", "channel")]
totalData[, by_ipchannel_prevclick1 := click_time_num - by_ipchannel_prevclick1]
totalData[is.na(by_ipchannel_prevclick1), by_ipchannel_prevclick1 := 0]
gc()

totalData[, by_ipos_prevclick1 := c(NA, click_time_num[-.N]), by = c("ip", "os")]
totalData[, by_ipos_prevclick1 := click_time_num - by_ipos_prevclick1]
totalData[is.na(by_ipos_prevclick1), by_ipos_prevclick1 := 0]
gc()

trainData <- totalData[1:endOfTrainData,]
testsupData <- totalData[(endOfTrainData+1):nrow(totalData),]

# rm(totalData)
# gc()

testData <- testData %>% left_join(testsupData, by = setdiff(colnames(testData),"click_id"))
rm(testsupData)
gc()


trainData$ip <- NULL
trainData$click_time <- NULL
trainData$click_time_num <- NULL
trainData$day <- NULL
gc()

testData$ip <- NULL
testData$click_time <- NULL
testData$click_time_num <- NULL
testData$day <- NULL
gc()


## factor변환

# trainDataDwnSmpl[, c("app","device","os","channel","is_attributed")] <- lapply(trainDataDwnSmpl[, c("app","device","os","channel","is_attributed")], as.factor)
# validData[, c("app","device","os","channel","is_attributed")] <- lapply(validData[, c("app","device","os","channel","is_attributed")], as.factor)
trainData[, c("app","device","os","channel","is_attributed")] <- lapply(trainData[, c("app","device","os","channel","is_attributed")], as.factor)
testData$is_attributed <- c(rep(1,(nrow(testData)+1)/2),rep(0,(nrow(testData)-1)/2))
testData[, c("app","device","os","channel","is_attributed")] <- lapply(testData[, c("app","device","os","channel","is_attributed")], as.factor)
gc()


## fst로 저장
write_fst(trainData, "/home/rjs/바탕화면/adtrack/data/trainData.fst")
write_fst(testData, "/home/rjs/바탕화면/adtrack/data/testData.fst")
gc()

rm(trainData,testData)

## fst load
trainData <- fst("/home/rjs/바탕화면/adtrack/data/trainData.fst")
testData <- fst("/home/rjs/바탕화면/adtrack/data/testData.fst")

## Modeling

predictors <- setdiff(names(trainData),
                      c("is_attributed"))
response <- "is_attributed"
gc()

## h2o
h2o.init(nthreads = 10, max_mem_size = '60G', enable_assertions = FALSE)
train_Hex<-as.h2o(trainData)
test_Hex<-as.h2o(testData)
gc()

rm(trainData, testData)
gc()

## Modeling

predictors <- setdiff(names(train_Hex),
                      c("is_attributed","attributed_time","click_time"))
response <- "is_attributed"


## gbm

gbm_all <- h2o.gbm(x = predictors,
                   y = response,
                   training_frame = train_Hex,
                   # validation_frame = val_Hex,
                   nfolds = 3,
                   ntrees = 100, 
                   sample_rate = 0.7, 
                   col_sample_rate = 0.7,
                   max_depth = 5, 
                   min_rows = 1,
                   seed=950902)

pred_gbm <- h2o.predict(gbm_all,test_Hex)

sub <- as.data.frame(x = testData$click_id)
sub$gbm_all <- as.vector(pred_gbm$predict)

colnames(sub) <- c("click_id","is_attributed")
write.csv(sub,paste0("/home/rjs/바탕화면/adtrack/result/gbm_all",Sys.Date(),".csv"),row.names=FALSE)

gbm_all

gbm_all@model$variable_importances[1:20,]




## xgboost

xgboost_all <- h2o.xgboost(x = predictors,
                           y = response,
                           training_frame = train_Hex,
                           # validation_frame = val_Hex,
                           nfolds = 3,
                           booster = "gbtree",
                           learn_rate = 0.05,
                           max_depth = 5,
                           subsample = 0.7,
                           col_sample_rate_per_tree = 0,
                           min_child_weight = 1, 
                           min_split_improvement = 0,
                           reg_lambda = 0, 
                           reg_alpha = 0,
                           seed=950902)




pred_xgboost <- h2o.predict(xgboost_all,test_Hex)

sub <- as.data.frame(x = testData$click_id)
sub$xgboost_all <- as.vector(pred_xgboost$predict)

colnames(sub) <- c("click_id","is_attributed")
write.csv(sub,paste0("/home/rjs/바탕화면/adtrack/result/xgboost_all",Sys.Date(),".csv"),row.names=FALSE)

xgboost_all

xgboost_all@model$variable_importances[1:20,]



## rf

rf_all <- h2o.randomForest(x = predictors,
                           y = response,
                           training_frame = train_Hex,
                           # validation_frame = val_Hex,
                           nfolds = 3,
                           col_sample_rate_change_per_level = 0.7, 
                           col_sample_rate_per_tree = 0.7,
                           ntree=100,
                           seed=950902)



pred_rf <- h2o.predict(rf_all,test_Hex)

sub <- as.data.frame(x = testData$click_id)
sub$rf_all <- as.vector(pred_rf$predict)

colnames(sub) <- c("click_id","is_attributed")
write.csv(sub,paste0("/home/rjs/바탕화면/adtrack/result/rf_all",Sys.Date(),".csv"),row.names=FALSE)


rf_all

rf_all@model$variable_importances[1:20,]



## glm

glm_all <- h2o.glm(x = predictors,
                   y = response,
                   training_frame = train_Hex,
                   # validation_frame = val_Hex,
                   seed=950902)



pred_glm <- h2o.predict(glm_all,test_Hex)

sub <- as.data.frame(x = testData$click_id)
sub$glm_all <- as.vector(pred_glm$predict)

colnames(sub) <- c("click_id","is_attributed")
write.csv(sub,paste0("/home/rjs/바탕화면/adtrack/result/glm_all",Sys.Date(),".csv"),row.names=FALSE)


glm_all

glm_all@model$variable_importances[1:20,]


## lightgbm

lightgbm_all <- h2o.xgboost(x = predictors,
                            y = response,
                            training_frame = train_Hex,
                            # validation_frame = val_Hex,
                            nfolds = 3,
                            tree_method="hist",
                            grow_policy="lossguide",
                            booster = "gbtree",
                            learn_rate = 0.05,
                            max_leaves = 255,
                            max_depth = 5,
                            max_bins = 100,
                            subsample = 0.7,
                            col_sample_rate_per_tree = 0,
                            min_child_weight = 1, 
                            min_split_improvement = 0,
                            reg_lambda = 0, 
                            reg_alpha = 0,
                            seed=950902)


pred_lightgbm <- h2o.predict(lightgbm_all,test_Hex)

sub <- as.data.frame(x = testData$click_id)
sub$lightgbm_all <- as.vector(pred_lightgbm$predict)

colnames(sub) <- c("click_id","is_attributed")
write.csv(sub,paste0("/home/rjs/바탕화면/adtrack/result/lightgbm_all",Sys.Date(),".csv"),row.names=FALSE)

lightgbm_all

lightgbm_all@model$variable_importances[1:20,]



## dart

dart_all <- h2o.xgboost(x = predictors,
                        y = response,
                        booster="dart",
                        rate_drop = 0.1,
                        training_frame = train_Hex,
                        # validation_frame = val_Hex,
                        nfolds = 3,
                        ntree=200,
                        seed=950902)





pred_dart <- h2o.predict(dart_all,test_Hex)

sub <- as.data.frame(x = testData$click_id)
sub$dart_all <- as.vector(pred_dart$predict)

colnames(sub) <- c("click_id","is_attributed")
write.csv(sub,paste0("/home/rjs/바탕화면/adtrack/result/dart_all",Sys.Date(),".csv"),row.names=FALSE)

dart_all

dart_all@model$variable_importances[1:20,]


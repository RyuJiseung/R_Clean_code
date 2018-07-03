val_ratio <- 0.90

train_col_names <- c("ip", "app", "device", "os", "channel", 
                     "click_time", "attributed_time", "is_attributed")

total_rows <- 184903890

#######################################################
library(data.table)
library(caret)
library(lubridate)
library(dplyr)

# Nos quedamos solo con las primeras 80000000 lineas de train.csv para no agotar la memoria
# Descartamos las columnas "ip" (1), "click time" (6) y "attributed time" (7)
train<-fread('/home/rjs/바탕화면/adtrack/data/train.csv',drop = c(7))
train$click_time<- ymd_hms(train$click_time)
train$click_time <- train$click_time + dhours(8)
train$hour <- hour(train$click_time)
train$day <- day(train$click_time)

testsupData <- fread('/home/rjs/바탕화면/adtrack/data/test_supplement.csv')
testsupData <- testsupData %>% select(-click_id)
testsupData <- testsupData[!duplicated(testsupData),]
testsupData$is_attributed <- 0
testsupData$click_time<-ymd_hms(testsupData$click_time)
testsupData$click_time <- testsupData$click_time + dhours(8)
testsupData$hour <- hour(testsupData$click_time)
testsupData$day <- day(testsupData$click_time)


# Descartamos las columnas "ip" (2) y "click time" (7)
test<-fread('/home/rjs/바탕화면/adtrack/data/test.csv')
test$click_time<-ymd_hms(test$click_time)
test$click_time <- test$click_time + dhours(8)

endOftrain<-dim(train)[1]

totalData <- rbind(train, testsupData)

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

totalData[, click_time_num := as.numeric(click_time)]

totalData[, by_ipappdeviceoschannel_nextclick1 := c(click_time_num[-1], NA), by = c("ip", "app", "device", "os", "channel")]
totalData[, by_ipappdeviceoschannel_nextclick1 := by_ipappdeviceoschannel_nextclick1 - click_time_num]
totalData[is.na(by_ipappdeviceoschannel_nextclick1), by_ipappdeviceoschannel_nextclick1 := 0]

totalData[, by_iposdevice_nextclick1 := c(click_time_num[-1], NA), by = c("ip", "os", "device")]
totalData[, by_iposdevice_nextclick1 := by_iposdevice_nextclick1 - click_time_num]
totalData[is.na(by_iposdevice_nextclick1), by_iposdevice_nextclick1 := 0]

totalData[, by_iposdeviceapp_nextclick1 := c(click_time_num[-1], NA), by = c("ip", "os", "device", "app")]
totalData[, by_iposdeviceapp_nextclick1 := by_iposdeviceapp_nextclick1 - click_time_num]
totalData[is.na(by_iposdeviceapp_nextclick1), by_iposdeviceapp_nextclick1 := 0]

totalData[, by_ipchannel_prevclick1 := c(NA, click_time_num[-.N]), by = c("ip", "channel")]
totalData[, by_ipchannel_prevclick1 := click_time_num - by_ipchannel_prevclick1]
totalData[is.na(by_ipchannel_prevclick1), by_ipchannel_prevclick1 := 0]

totalData[, by_ipos_prevclick1 := c(NA, click_time_num[-.N]), by = c("ip", "os")]
totalData[, by_ipos_prevclick1 := click_time_num - by_ipos_prevclick1]
totalData[is.na(by_ipos_prevclick1), by_ipos_prevclick1 := 0]

train <- totalData[1:endOftrain,]
testsupData <- totalData[(endOftrain+1):nrow(totalData),]

rm(totalData)
gc()

test <- test %>% left_join(testsupData, by = setdiff(colnames(test),"click_id"))
rm(testsupData)
gc()

# Como mas abajo hacemos rbind de train y z vamos a almacenar
# donde terminan los datos de trainig
train$ip <- NULL
train$click_time <- NULL
train$click_time_num <- NULL
train$day <- NULL

test$ip <- NULL
test$click_time <- NULL
test$click_time_num <- NULL
test$day <- NULL
gc()

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

rm(train.index, train)
invisible(gc())

#######################################################

cat("train size : ", dim(dtrain), "\n")
cat("valid size : ", dim(valid), "\n")

#######################################################

categorical_features = c("app", "device", "os", "channel", "hour")

#######################################################

# install.packages("h2o", repos=(c("http://s3.amazonaws.com/h2o-release/h2o/master/1497/R", getOption("repos"))))
library(h2o)
h2o.shutdown()
h2o.init(nthreads =- -1, max_mem_size = '50G')

dtrain[,c("is_attributed","app","os","device","channel")] <-lapply(dtrain[,c("is_attributed","app","os","device","channel")],as.factor)
gc()
train_Hex<-as.h2o(dtrain)
rm(dtrain);gc();

dtrain[,c("is_attributed","app","os","device","channel")] <-lapply(dtrain[,c("is_attributed","app","os","device","channel")],as.factor)
gc()
val_Hex<-as.h2o(valid)
rm(valid);gc();

test$is_attributed <- c(rep(1,(nrow(test)+1)/2),rep(0,(nrow(test)-1)/2))
test[,c("is_attributed","app","os","device","channel")] <-lapply(test[,c("is_attributed","app","os","device","channel")],as.factor)
gc()
test_Hex<-as.h2o(test)

predictors <- setdiff(names(test),
                      c("is_attributed", "click_id"))

response <- "is_attributed"


gbm_model <- h2o.gbm(x = predictors,
                     y = response,
                     training_frame = train_Hex,
                     validation_frame = val_Hex,
                     # categorical_encoding = "OneHotInternal",
                     # max_depth = 7,
                     # learn_rate = 0.1,
                     ntree = 1000,
                     # sample_rate = 0.7,
                     # col_sample_rate = 0.7,
                     # col_sample_rate_per_tree  = 0.8,
                     # tree_method="hist",
                     # grow_policy="lossguide",
                     # max_bin = 256,
                     seed=950902)

pred_gbm <- h2o.predict(gbm_model,test_Hex)

test$gbm <- as.vector(pred_gbm$p1)

temp <- test %>% select(click_id,gbm)
colnames(temp) <- c("click_id","is_attributed")
write.csv(temp,paste0("/home/rjs/바탕화면/adtrack/result/gbm_only_",sys.Date(),".csv"),row.names=FALSE)


xgb_model <- h2o.xgboost(x = predictors,
               y = response,
               training_frame = train_Hex,
               validation_frame = val_Hex,
               # categorical_encoding = "OneHotInternal",
               # max_depth = 7,
               # learn_rate = 0.1,
               ntree = 1000,
               # sample_rate = 0.7,
               # col_sample_rate = 0.7,
               # col_sample_rate_per_tree  = 0.8,
               # tree_method="hist",
               # grow_policy="lossguide",
               # max_bin = 256,
               seed=950902)

pred_xgb <- h2o.predict(xgb_model,test_Hex)

test$xgb <- as.vector(pred_xgb$p1)

temp <- test %>% select(click_id,xgb)
colnames(temp) <- c("click_id","is_attributed")
write.csv(temp,paste0("/home/rjs/바탕화면/adtrack/result/xgb_only_",sys.Date(),".csv"),row.names=FALSE)

gbm@model$variable_importances[1:20,]


rf
rf <- h2o.randomForest(x = predictors,
                       y = response,
                       training_frame = train_Hex,
                       validation_frame = val_Hex,
                       ntree=200,
                       seed=950902)


pred_rf <- h2o.predict(rf,test_Hex)

test_df$rf <- as.vector(pred_rf$p1)

temp <- test_df %>% select(msno,rf)
colnames(temp) <- c("msno","is_churn")
write.csv(temp,"D:/자료실/내 문서/OneDrive - 이화여자대학교/jiseung1216/머신러닝/프로젝트/캐글/wsdm/data/rf_only_201604.csv",row.names=FALSE)

rf@model$variable_importances[1:20,]
temp <- test_df %>% select(msno, gbm, rf)
temp$is_churn <- temp %>% with((gbm + rf)/2)
temp <- temp %>% select(msno, is_churn)
write.csv(temp,"D:/자료실/내 문서/OneDrive - 이화여자대학교/jiseung1216/머신러닝/프로젝트/캐글/wsdm/data/total_only_201604.csv",row.names=FALSE)
temp <- test_df %>% select(msno, gbm, rf)
temp$is_churn <- temp %>% with((gbm*rf)^(1/2))
temp <- temp %>% select(msno, is_churn)
write.csv(temp,"D:/자료실/내 문서/OneDrive - 이화여자대학교/jiseung1216/머신러닝/프로젝트/캐글/wsdm/data/Geo_total_only_201604.csv",row.names=FALSE)
temp <- test_df %>% select(msno, gbm, rf)
temp$is_churn <- temp %>% with(2/(1/gbm+1/rf))
temp <- temp %>% select(msno, is_churn)
write.csv(temp,"D:/자료실/내 문서/OneDrive - 이화여자대학교/jiseung1216/머신러닝/프로젝트/캐글/wsdm/data/harmonic_total_only_201604.csv",row.names=FALSE)


#*****************************************************************
# Preparing the sub data

cat("Setting up the submission file... \n")
sub <- data.table(click_id = test$click_id, is_attributed = NA) 
test$click_id <- NULL
invisible(gc())

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

#*****************************************************************
# Feature importance

#######################################################

cat("Feature importance: ")
klgb.importance(model, percentage = TRUE)

#######################################################

cat("\nAll done!..")

#######################################################

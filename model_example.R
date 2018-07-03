## AutoML ##
library(needs)
needs(dplyr,readr,h2o,stringr)

setwd("/Users/jinseokryu/Desktop/ML강의자료/skt/4_20180626/")

df = read.csv("horse_race.csv", stringsAsFactors = FALSE)


df$trRcCntT <- as.integer(str_replace(df$trRcCntT, ",", ""))
df$weather <- as.factor(df$weather)
df$track <- as.factor(df$track)
df$hrSex <- as.factor(df$hrSex)
df$rcType <- as.factor(df$rcType)
df$rcClass <- as.factor(df$rcClass)
df$rcResult <- factor(df$rcResult,levels=c("fail","pass"),labels=c(0,1))


x <- setdiff(colnames(df),"rcResult")
x <- c(setdiff(coef (cv.out, s = lambda_1se)@Dimnames[[1]][coef (cv.out, s = lambda_1se)@i],c("(Intercept)","hrSexstallion")),"hrSex")
y <- "rcResult"
  
  
h2o.init(nthreads =- -1, max_mem_size = '8G')

df_h2o <- as.h2o(df)

h2o_split <- h2o.splitFrame(df_h2o,ratios = c(0.5,0.3))

train <- h2o_split[[1]]
valid <- h2o_split[[2]]
test <- h2o_split[[3]]

aml1 <- h2o.automl(x = x, y = y,
                  training_frame = train,
                  validation_frame = valid,
                  leaderboard_frame = test,
                  stopping_rounds = 10,
                  stopping_tolerance = 0.01,
                  max_runtime_secs = 60*1*1)

aml2 <- h2o.automl(x = x, y = y,
                   training_frame = train,
                   validation_frame = valid,
                   leaderboard_frame = test,
                   stopping_rounds = 10,
                   stopping_tolerance = 0.1,
                   max_runtime_secs = 60*30)

light_xgb <- h2o.xgboost(x = x, y = y,
                         training_frame = df,
                         nfolds = 5,
                         ntrees = 50,
                         tree_method="hist", 
                         grow_policy="lossguide",
                         stopping_metric = "logloss",
                         stopping_rounds = 10,
                         stopping_tolerance = 0.05)

aml2@leaderboard

aml1@leader

glm_model <- h2o.glm(x = x, y = y,
                     training_frame = train,
                     validation_frame = valid,
                     solver = "IRLSM", standardize = T, link = "logit",
                     family = "binomial", alpha = 0.5, lambda = 1e-05)

auc <- h2o.auc(object = glm_model)
print(paste0("AUC of the training set : ", round(auc, 4)))
print(glm_model@model$standardized_coefficient_magnitudes)
print(glm_model@model$scoring_history)

### Building a Deep Learning Model
###### Build a binary classfication model using function `h2o.deeplearning` and selecting “bernoulli” for parameter `Distribution`.
###### Run 100 passes over the data by setting parameter `epoch` to 100.

dl_model <- h2o.deeplearning(x = x, y = y,
                             training_frame = train,
                             validation_frame = valid, distribution = "bernoulli", 
                             epochs = 100, hidden = c(200,200), target_ratio_comm_to_comp = 0.02, seed = 6765686131094811000, variable_importances = T)
auc2 <- h2o.auc(object = dl_model)
print(paste0("AUC of the training set : ", round(auc2, 4)))
print(h2o.varimp(dl_model))
print(h2o.scoreHistory(dl_model))

### All done, shutdown H2O    
h2o.shutdown(prompt=FALSE)



## ridge_logistic_regression ##
#set seed to ensure reproducible results
set.seed(42)
#split into training and test sets
df[,"train"] <- ifelse(runif(nrow(df))<0.8,1,0)
#separate training and test sets
trainset <- df[df$train==1,]
testset <- df[df$train==0,]
#get column index of train flag
trainColNum <- grep("train",names(trainset))
#remove train flag column from train and test sets
trainset <- trainset[,-trainColNum]
testset <- testset[,-trainColNum]
#get column index of predicted variable in dataset
typeColNum <- grep("rcResult",names(df))
#build model
glm_model <- glm(rcResult~.,data = trainset, family = binomial)
summary(glm_model)

#load required library
library(glmnet)
#convert training data to matrix format
x <- model.matrix(rcResult~.,trainset)
#convert class to numerical variable
y <- ifelse(trainset$rcResult==1,1,0)
#perform grid search to find optimal value of lambda
#family= binomial => logistic regression, alpha=1 => lasso
# check docs to explore other type.measure options
cv.out <- cv.glmnet(x,y,alpha=1,family="binomial",type.measure = "mse" )
#plot result
plot(cv.out)

#min 람다 값
lambda_min <- cv.out $ lambda.min
#best 값 람다
lambda_1se <- cv.out $ lambda.1se
# 계수 계수
coef (cv.out, s = lambda_1se)

coef (cv.out, s = lambda_1se)@Dimnames[[1]][coef (cv.out, s = lambda_1se)@i]

#get test data
x_test <- model.matrix(rcResult~.,testset)
#predict class, type=”class”
lasso_prob <- predict(cv.out,newx = x_test,s=lambda_1se,type="response")
#translate probabilities to predictions
lasso_predict <- rep(0,nrow(testset))
lasso_predict[lasso_prob>.5] <- 1
#confusion matrix
table(pred=lasso_predict,true=testset$rcResult)

LogLoss<-function(actual, predicted)
{
  result<- -1/length(actual)*(sum((actual*log(predicted)+(1-actual)*log(1-predicted))))
  return(result)
}
LogLoss(actual=as.numeric(as.character(testset$rcResult)), predicted=as.vector(lasso_prob))

#accuracy
mean(lasso_predict==testset$rcResult)

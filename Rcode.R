library(mlr)
library(dplyr)
library(ggplot2)
library(xgboost)
rm(list = ls())
set.seed(123)
setwd("/home/yan123/module3")

# "XTR.txt" and "XTE.txt" were generated from "Xtrain.txt" and "Xtest.txt"
# with "#" and "?" removed using terminal command
X = read.table("XTR.txt", sep=' ', header = T, row.names = NULL)
Y = read.csv("Ytrain.txt", header = T, row.names = NULL)
XT = read.table("XTE.txt", sep=' ', header = T, row.names = NULL)


# remove columns containing only constant values in the test set
constV = names(which(sapply(XT,function(x){var(x,na.rm = T)==0})==T))
XT = XT[,!colnames(XT)%in%constV]
# extract Id in the test set
XTid = XT$Id
XT = subset(XT, select = -Id)
# replace -9 with NA in the test set
XT[!is.na(XT)&(XT<=-8.9)] = NA


# bind reponse varible to the train set
samples = cbind(X,Value = Y[,"Value"])
# remove the constant columns of test set in the train set
samples = samples[,!colnames(samples)%in%constV]
# extract id in the train set
samples = subset(samples, select = -Id)
# replace -9 with NA in the train set
samples[!is.na(samples)&(samples<=-8.9)] = NA

# remove unusual 7 in the train set
all7TR = apply(samples[,!colnames(samples)%in%c("Value","G01","G02","G03")],1,function(a){T%in%(a>6.9)})
samples = samples[!all7TR,]

# remove unusual 7 in the test set
all7TE = apply(XT[,!colnames(XT)%in%c("Value","G01","G02","G03")],1,function(a){T%in%(a>6.9)})
XT_n7 = XT[!all7TE,]

# using a portion of train set for training, here is 100%
S_index = sample(nrow(samples),1*floor(nrow(samples)))
SS = samples[S_index,]

# % of rows with missing value
sum(apply(SS,1,function(x){F%in%is.na(x)}))/nrow(SS)

# split train set further into test and train set
# to satisfy the format of the input of the preprocess function below
index = sample(nrow(SS),floor(0.75*nrow(SS)))
train = SS[index,]
test = SS[-index,]

# function to preprocess the data
data_prep = function(train, test, option) {
  
  if("Value"%in%colnames(test)) {
    test_y = test[,"Value"]
    test = test[,!colnames(test)%in%"Value"]
  }
  if("Value"%in%colnames(train)) {
    train_y = train[,"Value"]
    train = train[,!colnames(train)%in%"Value"]
  }
  
  n_train = nrow(train)
  n_test = nrow(test)
  n_tot = n_train+n_test
  
  tot = rbind(train,test)
  
  # augmentation by adding missing indicators
  imp = sapply(tot,function(x){as.numeric(is.na(x))})
  colnames(imp) = paste0("V",seq(1:ncol(imp)))
  
  # impute missing values with column median
  tot = randomForest::na.roughfix(tot)
  
  tot = cbind(tot,imp)
  
  # remove constant columns if any
  tot = tot[,apply(tot, 2, var, na.rm=TRUE) != 0]
  
  train = cbind(tot[1:n_train,],"Value" = train_y)
  
  if(option == 0){
    
    test = cbind(tot[(n_train+1):n_tot, ], "Value" = test_y)
    tot = rbind(train,test)
    return(tot)
    
  } else if(option == 1) {
    
    test = tot[(n_train+1):n_tot, ]
    return(list(train,test))
    
  }
}

# transform the train set using the preprocess function defined above
tot = data_prep(train = train, test = test,option = 0)

# define train task for 'mlr' package
tsk = makeRegrTask(data = tot, target = "Value")

# split data into train and validation for 'mlr'
h = makeResampleDesc("Holdout")
ho = makeResampleInstance(h,tsk)
tsk.train = subsetTask(tsk,ho$train.inds[[1]])
tsk.test = subsetTask(tsk,ho$test.inds[[1]])

#------------------ xgboost ------------------

# determine the appropriate values for hyperparameter 'eta' and 'nrounds' with cross-validation 
xgb_train = as.matrix(tot)
dtrain = xgb.DMatrix(data = subset(xgb_train,select = -Value),label = subset(xgb_train,select = Value))
params <- list(booster = "gbtree",
               objective = "reg:linear", eta=0.1, gamma=0, max_depth=3, min_child_weight=7, subsample=0.9, colsample_bytree=1)
xgbcv <- xgb.cv( params = params, data =dtrain, nrounds = 2000, nfold = 3, showsd = F,
                 print_every_n = 40, early_stop_round = 20, maximize = F,metrics = 'rmse')


# train the model with train set (not the entire train set, but a subset)
xgb_lrn = makeLearner(cl = "regr.xgboost",predict.type = "response")
xgb_lrn$par.vals = list(booster = "gbtree", objective = "reg:linear", eta=0.1, gamma=0, max_depth=3, min_child_weight=7,
                        subsample=0.9, colsample_bytree=1,nrounds = 5000,print_every_n = 40)

# validate the model using validation set(generated from the train set)
xgb_mod = train(xgb_lrn, tsk.train)
xgb_pred = predict(xgb_mod, tsk.test)
performance(xgb_pred, measures = rmse)

#------------------------------------------------


# generate the part of the sumbission file
# for rows in the test set with unusal 7
submission7 = data.frame(matrix(NA,nrow=sum(all7TE),ncol = 2))
colnames(submission7) = c("Id","Value")
submission7$Id = XTid[all7TE]
submission7$Value = 50.17517 

# preprocess the test set without rows with unusual 7 
sub = data_prep(train = SS, test = XT_n7,option = 1)
sub = sub[[2]]


# define function for generate submission file
make_prediction = function(lrn,tsk,sub_data,subname) {
  
  # train a model
  mod = train(lrn,tsk)
  # make predictions using test data
  pred = predict(mod,newdata = sub_data)
  
  # generate a submission file and bind it with another part of the submission file created above 
  submission = data.frame(matrix(NA,nrow=sum(!all7TE),ncol = 2))
  colnames(submission) = c("Id","Value")
  submission$Id = XTid[!all7TE]
  submission$Value = pred$data$response
  submission = arrange(rbind(submission,submission7),by = Id)
  write.csv(submission,file = subname,row.names = F, col.names = T)
  
}

# train a model using the entire training set and generate the submission file using the above function
make_prediction(lrn = xgb_lrn,tsk = tsk,sub_data = sub,subname = "xgb_basic2.csv")
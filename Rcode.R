library(mlr)
library(dplyr)
library(ggplot2)
library(xgboost)
source("D:/sfu/kaggle/b/b/fs.R")

rm(list = ls())
set.seed(123)
setwd("D:/sfu/stat 440/module3")

X = read.table("XTR.txt", sep=' ', header = T, row.names = NULL)
Y = read.csv("Ytrain.txt", header = T, row.names = NULL)
XT = read.table("XTE.txt", sep=' ', header = T, row.names = NULL)

for(i in 1:ncol(XT)){
  hist(XT[,i],breaks = 100, main = names(XT)[i])
}
constV = names(which(sapply(XT,function(x){var(x,na.rm = T)==0})==T))


XT = XT[,!colnames(XT)%in%constV]
XTid = XT$Id
XT = subset(XT, select = -Id)
XT[!is.na(XT)&(XT<=-0.89)] = NA

  
# problematic variables c03 b23 b20 b15 b14 b13 b12 b11
samples = cbind(X,Value = Y[,"Value"])
samples = samples[,!colnames(samples)%in%constV]
samples = subset(samples, select = -Id)

samples[!is.na(samples)&(samples<=-0.89)] = NA



missing = as.vector(NULL)
for(i in 1:ncol(samples)) {
  n = sum(is.na(samples[,i]))
  p = round(n/nrow(samples),2)
  if(n > 0) {
    print(paste0(colnames(samples[i]),": ",n," ",p))
    missing = c(missing,colnames(samples[i]))
  }
}

for(i in 1:ncol(samples)){
  hist(samples[,i],breaks = 100,main = names(samples)[i])
}


# G6 = apply(samples[,!colnames(samples)%in%"Value"],1,function(x){T%in%(x>6)})
# plot(density(samples[G6,"Value"]))

S_index = sample(nrow(samples),floor(0.03*nrow(samples)))
SS = samples[S_index,]

# observing missing value in each row
# knn is not applicable (at least with 3% of data)
sum(apply(SS,1,function(x){F%in%is.na(x)}))/nrow(SS)

index = sample(nrow(SS),floor(0.75*nrow(SS)))
train = SS[index,]
test = SS[-index,]


# p = ggplot(samples, aes(x=X.H08, y=Value)) +
#   ggtitle("scatter plot") + xlab("x") + ylab("Value")
# p1 = p + geom_point(alpha = 0.01, colour = "orange") +
#   geom_density2d() + theme_bw()
# 
# 
# p2 = p + stat_bin_hex(colour="white", na.rm=TRUE) +
#   scale_fill_gradientn(colours=c("purple","green"), 
#                        name = "Frequency", 
#                        na.value=NA)


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

  ### remove outliers
  # ...
  
    
  # prepro = caret::preProcess(tot,method = "bagImpute")
  # tot = predict(prepro,tot)
  
  tot = randomForest::na.roughfix(tot)
  
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



tot = data_prep(train = train, test = test,option = 0)

for(i in 1:ncol(tot)){
  hist(tot[,i],breaks = 100,main = names(tot)[i])
}
# write.csv(tot, file = "bagImpute.csv",row.names = F, col.names = T)

tsk = makeRegrTask(data = tot, target = "Value")

# split data into train and test
h = makeResampleDesc("Holdout")
ho = makeResampleInstance(h,tsk)
tsk.train = subsetTask(tsk,ho$train.inds[[1]])
tsk.test = subsetTask(tsk,ho$test.inds[[1]])

# use all cpus during training
library(parallel)
library(parallelMap)
parallelStartSocket(cpus = detectCores()-1)

# number of iterations used for hyperparameters tuning
tc = makeTuneControlRandom(maxit = 30)

# resampling strategy for evaluating model performance
rdesc = makeResampleDesc("CV", iters = 3)


#------------------ randomForest ------------------
# # build model
# rf_lrn = makeLearner(cl ="regr.randomForest", par.vals = list())
# # define the search range of hyperparameters
# rf_ps = makeParamSet( makeIntegerParam("ntree",150,600),makeIntegerParam("nodesize",lower = 3,upper = 15),
#                       makeIntegerParam("se.ntree",lower = 50 ,upper = 300), makeIntegerParam("se.boot",lower = 50,upper = 300),
#                       makeIntegerParam("mtry",lower = 20,upper = 60),makeLogicalParam("importance",default = FALSE))
# 
# # search for the best hyperparameters
# rf_tr = tuneParams(rf_lrn,tsk.train,cv3,rmse,rf_ps,tc)
# # specify the hyperparmeters for the model
# rf_lrn = setHyperPars(rf_lrn,par.vals = rf_tr$x)
# rf_mod = train(rf_lrn, tsk.train)
# # evaluate performance use CV
# r = resample(rf_lrn, tsk, resampling = rdesc, show.info = T, models = FALSE,measures = rmse)
# plotFeatureImportance(rf_mod,10)
#------------------------------------

#------------------ mars ------------------
# mars_lrn = makeLearner(cl = "regr.mars",par.vals = list())
# mars_lrn = makePreprocWrapperCaret(mars_lrn, ppc.pca = TRUE, ppc.pcaComp = 20)
# mars_ps = makeParamSet( makeNumericParam("thresh",lower = 0.0001, upper= 0.01),makeNumericParam("penalty",lower = 1,upper = 4),
#                        makeIntegerParam("degree",lower = 1,upper = 4))
# mars_tr = tuneParams(mars_lrn,tsk.train,cv3,rmse,mars_ps,tc)
# mars_lrn = setHyperPars(mars_lrn,par.vals = mars_tr$x)
# 
# # mars_mod = train(mars_lrn, tsk.train)
# # mars_pred = predict(mars_mod, tsk.test)
# r = resample(mars_lrn, tsk, resampling = rdesc, show.info = T, models = T,measures = rmse)
#------------------------------------

#------------------ xgboost ------------------

xgb_train = as.matrix(tot[1:nrow(train),])
xgb_test = as.matrix(tot[(nrow(train)+1):nrow(SS),])
dtrain = xgb.DMatrix(data = subset(xgb_train,select = -Value),label = subset(xgb_train,select = Value)) 
dtest = xgb.DMatrix(data =  subset(xgb_test,select = -Value),label= subset(xgb_test,select = Value))

params <- list(booster = "gbtree",
               objective = "reg:linear", eta=0.05, gamma=0, max_depth=6, min_child_weight=1, subsample=0.8, colsample_bytree=0.8,nthread = 8)
xgbcv <- xgb.cv( params = params, data =dtrain, nrounds = 500, nfold = 5, showsd = T, 
                 print_every_n = 10, early_stop_round = 20, maximize = F,metrics = 'rmse')

# tuning
xgb_lrn = makeLearner(cl = "regr.xgboost",predict.type = "response")
xgb_lrn$par.vals = list(objective="reg:linear", eval_metric="error", nrounds=300, eta=0.05,verbose=0)
xgb_ps = makeParamSet( makeIntegerParam("max_depth",lower = 3,upper = 6),
                       makeNumericParam("min_child_weight",lower = 1,upper = 9), makeNumericParam("subsample",lower = 0.5,upper = 1),
                       makeNumericParam("colsample_bytree",lower = 0.5,upper = 1))
xgb_tr = tuneParams(xgb_lrn,tsk.train,cv3,rmse,xgb_ps,tc)
xgb_lrn = setHyperPars(xgb_lrn,par.vals = xgb_tr$x)

xgb_mod = train(xgb_lrn, tsk.train)
# xgb_pred = predict(xgb_mod, tsk.test)
# performance(xgb_pred, measures = mae)
r = resample(xgb_lrn, tsk, resampling = rdesc, show.info = T, models = FALSE,measures = rmse)

#------------------------------------------------







sub = data_prep(train = SS, test = XT,option = 1)
sub = sub[[2]]

make_prediction = function(lrn,tsk,sub_data,subname) {
  mod = train(lrn,tsk)
  pred = predict(mod,newdata = sub_data)
  
  
  rdesc = makeResampleDesc("CV", iters = 3)
  r = resample(lrn, tsk, resampling = rdesc, show.info = T, models = FALSE,measures = rmse)
  
  submission = data.frame(matrix(NA,nrow=length(XTid),ncol = 2))
  colnames(submission) = c("Id","Value")
  submission$Id = XTid
  submission$Value = pred$data$response
  write.csv(submission,file = subname,row.names = F, col.names = T)
  
}

make_prediction(lrn = xgb_lrn,tsk = tsk,sub_data = sub,subname = "xgb.csv")



TRT = sample_n(samples[-S_index,],50000)
TRT_y = TRT$Value
TRS = data_prep(train = SS, test = TRT,option = 1)
TRS = TRS[[2]]

mod1 = train(xgb_lrn,tsk)
pred1 = predict(mod1,newdata = TRS)
P_y = pred1$data$response
RMSE = sqrt(mean((TRT_y-P_y)^2))
RMSE






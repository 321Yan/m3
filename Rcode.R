library(mlr)
library(dplyr)
library(ggplot2)


rm(list = ls())
setwd("D:/sfu/stat 440/module3")

X = read.csv("Xtrain.txt", sep=' ', row.names = NULL)
Y = read.csv("Ytrain.txt",  row.names = NULL)
samples = cbind(X,Value = Y[,"Value"])

missing = as.vector(NULL)
for(i in 1:ncol(samples)) {
  n = sum(is.na(samples[,i]))
  p = round(n/nrow(samples),2)
  if(n > 0) {
    print(paste0(colnames(samples[i]),": ",n," ",p))
    missing = c(missing,colnames(samples[i]))
  }
}


index = sample(nrow(samples),floor(0.75*nrow(samples)))
train = samples[index,]
test = samples[-index,]


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
  
  tot[,sapply(tot,is.factor)] = as.numeric(tot[,sapply(tot,is.factor)])

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
tc = makeTuneControlRandom(maxit = 1)

# resampling strategy for evaluating model performance
rdesc = makeResampleDesc("RepCV", reps = 2, folds = 3)


#------------------ randomForest ------------------
# build model
rf_lrn = makeLearner(cl ="regr.randomForest", par.vals = list())
# define the search range of hyperparameters
rf_ps = makeParamSet( makeIntegerParam("ntree",150,600),makeIntegerParam("nodesize",lower = 3,upper = 15),
                      makeIntegerParam("se.ntree",lower = 50 ,upper = 300), makeIntegerParam("se.boot",lower = 50,upper = 300),
                      makeIntegerParam("mtry",lower = 20,upper = 60),makeLogicalParam("importance",default = FALSE))

# search for the best hyperparameters
rf_tr = tuneParams(rf_lrn,tsk.train,cv3,rmse,rf_ps,tc)
# specify the hyperparmeters for the model
rf_lrn = setHyperPars(rf_lrn,par.vals = rf_tr$x)
detach(package:caret)

# evaluate performance use CV
r = resample(rf_lrn, tsk, resampling = rdesc, show.info = T, models = FALSE,measures = mae)



library(randomForest)
library(xgboost)
library(keras)
library(mlr)
library(dplyr)
library(ggplot2)

rm(list = ls())
set.seed(456)
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

S_index = sample(nrow(samples),floor(0.5*nrow(samples)))
SS = samples[S_index,]


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
  
  
  # augmentation
  imp = sapply(tot,function(x){as.numeric(is.na(x))})
  colnames(imp) = paste0("V",seq(1:ncol(imp)))
  
  
  # prepro = caret::preProcess(tot,method = "bagImpute")
  # tot = predict(prepro,tot)
  
  tot = randomForest::na.roughfix(tot)
  
  tot = cbind(tot,imp)
  
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


train_labels = as.matrix(train[,"Value"])
train_data = as.matrix((tot[1:nrow(train),!colnames(tot)%in%"Value"]))


test_data = as.matrix(tot[(nrow(train)+1):nrow(tot),!colnames(tot)%in%"Value"])

test_labels = as.matrix(test[,"Value"])


############## NN ##############
# layer_dense(units = 128, activation = "softmax",
#             input_shape = dim(train_data)[2]) %>%
#   #, activation = "relu"
#   layer_dropout(0.3) %>%
#   layer_dense(units = 128, activation = "softmax") %>%
#   layer_dropout(0.3) %>%
#   layer_dense(units = 128, activation = "softmax") %>%
#   layer_dropout(0.3) %>%
#   layer_dense(units = 1)
build_model <- function() {
  
  model <- keras_model_sequential() %>%
    layer_dense(units = 64, activation = "relu",
                input_shape = dim(train_data)[2]) %>%
    #, activation = "relu"
    layer_dropout(0.2) %>%
    layer_dense(units = 64, activation = "relu") %>%
    layer_dropout(0.2) %>%
    layer_dense(units = 64, activation = "relu") %>%
    layer_dropout(0.2) %>%
    layer_dense(units = 64, activation = "relu") %>%
    layer_dropout(0.2) %>%
    layer_dense(units = 64, activation = "relu") %>%
    layer_dropout(0.2) %>%
    layer_dense(units = 64, activation = "relu") %>%
    layer_dropout(0.2) %>%
    layer_dense(units = 1)
  
  model %>% compile(
    loss = "mse",
    # optimizer_adam(lr = 0.0001, decay = 0.00001)
    optimizer = optimizer_adam(lr = 0.005, decay = 0.000002)
  )
  
  model
}

model <- build_model()



epochs <- 100
# Fit the model and store training stats
history <- model %>% fit(
  train_data,
  train_labels,
  epochs = epochs,
  validation_split = 0.2,
  batch_size = 64,
  verbose = 1
)

plot(history, loss = "mse", smooth = FALSE) +
  coord_cartesian(ylim = c(0, 150))

# make prediction and compute rmse
test_predictions <- model %>% predict(test_data)

rmse = sqrt(mean((test_labels-test_predictions)^2))
rmse

# ### ensemble learning
# DNN = data.frame(test_predictions)
# write.csv(DNN,file = "DNN.csv",row.names = F, col.names = T)

##################################################################################
##################################################################################


sub = data_prep(train = SS, test = XT,option = 1)
sub = sub[[2]]



TEST_data = as.matrix(sub)

# make prediction and compute rmse
TEST_predictions <- model %>% predict(TEST_data)

submission = data.frame(matrix(NA,nrow=length(XTid),ncol = 2))
colnames(submission) = c("Id","Value")
submission$Id = XTid
submission$Value = TEST_predictions 
write.csv(submission,file = "DNN.csv",row.names = F, col.names = T)
#########################################################################



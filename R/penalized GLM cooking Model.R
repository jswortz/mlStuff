#install.packages("penalized")
#library("penalized")
LAUNDRY_TRAINING_1 <- read.csv("c:/users/jeremy/downloads/STD_COOK.csv")
####begin partitioning training data set - will use fold-over validation 
train = na.omit(LAUNDRY_TRAINING_1)
response <- train['TARGET']
covars <- subset(train, select = -c(TARGET,X_dataobs_))
#recoding of response varaible to numeric vector
y <- numeric()
index2=sample(1:nrow(covars),30000)
samp1 = data.matrix(covars[index2,])
sampy = as.vector(response[index2,])
y[sampy == "WHR"] <- 1
y[sampy == "MAY"] <- 2
y[sampy == "AMA"] <- 3
y[sampy == "KAD"] <- 4
y[sampy == "JEN"] <- 5
#multiple imputation load library for imputation below
library(glmnet)
iter=seq(.0,.1,by=.05)
stats=data.frame()
# model <- cv.glmnet(x=samp1,y,family="multinomial",alpha=5,type.measure="class",standardize=FALSE)

for(i in iter)
{
model <- cv.glmnet(x=samp1,y,family="multinomial",nfolds=30,alpha=i,type.measure="class",standardize=FALSE)
best.iter=match(model$lambda.min,model$lambda)
error=model$cvm[best.iter]
row=data.frame(lambda=model$lambda.min,alpha=i,error=error)
stats=rbind(stats,row)
name=paste("model",i*100,sep="")
assign(name,model)
}
#best fit is alpha = .2

glmnet.bootstrap = function(d,alpha,lambda)
{
  train = d
  response <- train['TARGET']
  covars <- subset(train, select = -c(TARGET,X_dataobs_))
  #recoding of response varaible to numeric vector
  y <- numeric()
  samp1 = data.matrix(covars)
  sampy = as.vector(response)
  y[sampy == "WHR"] <- 1
  y[sampy == "MAY"] <- 2
  y[sampy == "AMA"] <- 3
  y[sampy == "KAD"] <- 4
  y[sampy == "JEN"] <- 5
  model=glmnet(x=samp1,y,family="multinomial",alpha=alpha,lambda=lambda,standardize=FALSE)
  return(coef(model))
}
bootstrappin = boot(train,glmnet.bootstrap,l=1000,alpha=.5,lambda=.0002)


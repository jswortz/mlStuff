#install.packages("penalized")
#library("penalized")
LAUNDRY_TRAINING_2 <- read.csv("c:/users/jeremy/dropbox/glmnet work/LAUND_STD.csv")
####begin partitioning training data set - will use fold-over validation 
LAUNDRY_TRAINING_2 = na.omit(LAUNDRY_TRAINING_2)
indexes <- sample(1:nrow(LAUNDRY_TRAINING_2), size=.2*nrow(LAUNDRY_TRAINING_2))
test <- LAUNDRY_TRAINING_2[indexes,]
train <- LAUNDRY_TRAINING_2[-indexes,]
response <- train['TARGET']
covars <- subset(train, select = -c(TARGET))
#recoding of response varaible to numeric vector
y <- numeric()
index2=sample(1:116364,50000)
samp1 = data.matrix(covars[index2,])
sampy = as.vector(response[index2,])
y[sampy == "WHR"] <- 1
y[sampy == "MAY"] <- 2
y[sampy == "AMA"] <- 3
#multiple imputation load library for imputation below
library(Hmisc)
library(mi)
library(glmnet)
foldID=sample(0:9,50000,replace=TRUE)
iter=seq(0,.1,by=.05)
stats_laund=data.frame()
for(i in iter)
{
model <- cv.glmnet(x=samp1,y,family="multinomial",alpha=i,type.measure="class",standardize=FALSE)
best.iter=match(model$lambda.min,model$lambda)
error=model$cvm[best.iter]
row=data.frame(lambda=model$lambda.min,alpha=i,error=error)
stats_laund=rbind(stats_laund,row)
name=paste("model",i*100,sep="")
assign(name,model)
}
#best fit is alpha = .2
model1 <- cv.glmnet(x=samp1,y,family="multinomial",alpha=.8,standardize=FALSE,type.measure="class")
model2 <- cv.glmnet(x=samp1,y,family="multinomial",alpha=.5,standardize=FALSE,type.measure="class")
model3 <- cv.glmnet(x=samp1,y,family="multinomial",alpha=0,standardize=FALSE,type.measure="class")
model4 <- cv.glmnet(x=samp1,y,family="multinomial",alpha=1,standardize=FALSE,type.measure="class")
model5 <- cv.glmnet(x=samp1,y,family="multinomial",alpha=.2,standardize=FALSE,type.measure="class")
model6 <- cv.glmnet(x=samp1,y,family="multinomial",alpha=.7,standardize=FALSE,type.measure="class")
model7 <- cv.glmnet(x=samp1,y,family="multinomial",alpha=.9,standardize=FALSE,type.measure="class")
model8 <- cv.glmnet(x=samp1,y,family="multinomial",alpha=.3,standardize=FALSE,type.measure="class")

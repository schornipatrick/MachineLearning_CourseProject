# LOAD PACKAGES
library(AppliedPredictiveModeling); library(caret); library(ElemStatLearn); library(pgmm)
library(rpart); library(gbm); library(lubridate); library(forecast); library(e1071); library(elasticnet)
library(lattice); library(ggplot2); library(corrplot); library(plyr); library(randomForest)

# DOWNLOADING THE DATA

# download training data
trainfile <- "train.csv"
trainurl <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"

if (!exists(trainfile)) {
  download.file(url = trainurl, destfile = trainfile, method = "curl")
}

# download testing data
testfile <- "test.csv"
testurl <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"

if (!exists(testfile)) {
  download.file(url = testurl, destfile = testfile, method = "curl")
}

# reading in the data
training <- read.csv(trainfile)
testing <- read.csv(testfile)

# SPLIT IN MY TRAINING AND MY TEST SET, USING INITIAL TRAINING DATA
inTrain <- createDataPartition(y = training$classe, p = 0.7, list = FALSE)
myTrain <- training[inTrain, ]
myTest <- training[-inTrain, ]

# CLEAN DATA

# Change fields that include "#DIV/0!" with NA
for (i in 1:(ncol(myTrain)-1)) {
  myTrain[, i] <- revalue(x = as.character(myTrain[, i]), replace = c("#DIV/0!" = NA), warn_missing = FALSE)
}

# Remove columns that have more than X% NAs

X <- 0.8
miss <- c()

for (i in 1:ncol(myTrain)) {
  if (sum(is.na(myTrain[, i])) > X * nrow(myTrain)) {miss <- append(miss, i)}
}
myTrain <- myTrain[, -miss]

# Remove number, user name and time stamps and window
miss2 <- c(1:7)
myTrain <- myTrain[, -miss2]

# evaluating and cohercing data class

# We want to make sure, factors are numeric, except for the classe.
for (i in 1:(ncol(myTrain)-1)) {
  myTrain[,i] <- as.numeric(myTrain[,i])
}

# since there were many NAs introduced here, we can again remove columns that have more than X% NAs
miss <- c()

for (i in 1:ncol(myTrain)) {
  if (sum(is.na(myTrain[, i])) > X * nrow(myTrain)) {miss <- append(miss, i)}
}
myTrain <- myTrain[, -miss]

# After removing some columns, check for dependence among the remaining variables

# For the correlation matrix, remove variable I want to predict
corMyTrain <- subset(myTrain, select = -c(classe))

# plot dependence among variables
cor <- cor(corMyTrain)
corrplot(cor, method = "circle", tl.cex = 0.4)

# As we can identify some dark blue spots, highly correlated variables (cor > Y%) shall be removed
# We hope, this accelerates the model building.

Y <- 0.98
miss3 <- findCorrelation(cor(corMyTrain), cutoff = Y)
miss3 <- sort(miss3)
myTrain <- myTrain[, -miss3]

# illustrate new dependence
corMyTrain <- corMyTrain[, -miss3]
cor <- cor(corMyTrain)
corrplot(cor, method = "circle", tl.cex = 0.4)

# In total, we reduced the number of variables from 160 to 50, using X = 0.8 and Y = 0.98

# BUILD THE MODEL

# Use Random Forest because it proved the highest accuracy in class. 
# First, we check the accuracy using only the random forest. 
# Based on that we will decide whether we will apply some combining of different models.

ModRF <- randomForest(classe~., data = myTrain, ntree = 200) 
# number of trees is set to 200 (default is 500) to potentially reduce computation time.
# While the random forest using caret (method = "rf") and default settings 
# did not complete computation in hours, this model could be built in a matter of minutes

# Examine performance on myTraining set

TrainPred <- predict(ModRF, myTrain)
confMatTrain <- confusionMatrix(TrainPred, myTrain$classe)
confMatTrain
table(TrainPred, myTrain$classe)
# this gives us confidence to apply the random forest model on the myTest set.

# Examine performance on myTest set
TestPred <- predict(ModRF, myTest)
confMatTest <- confusionMatrix(TestPred, myTest$classe)
confMatTest
table(TestPred, myTest$classe)

# Solving the quiz with the testing set
QuizPred <- predict(ModRF, testing)
QuizPred
# BOOM. Achieved 20/20 points.





# Set working directory to the folder file is contained in
this.dir <- dirname(parent.frame(2)$ofile)
setwd(this.dir)

set.seed(1234)
# Load relevant libraries
library(knitr)
library(markdown)
library(caret)
library(data.table)

trainUrl <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
testUrl <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"



# Read data 
trainRaw <- read.csv("./data/pml-training.csv")
testRaw <- read.csv("./data/pml-testing.csv")

# Checking Dimensions of Data
dim(trainRaw)
dim(testRaw)

#Check for NA values
sum(complete.cases(trainRaw))
sum(complete.cases(testRaw))

# Remove NA values
trainRaw <- trainRaw[, colSums(is.na(trainRaw)) == 0] 
testRaw <- testRaw[, colSums(is.na(testRaw)) == 0] 


classe <- trainRaw$classe
trainRemove <- grepl("^X|timestamp|window", names(trainRaw))
trainRaw <- trainRaw[, !trainRemove]
trainCleaned <- trainRaw[, sapply(trainRaw, is.numeric)]
trainCleaned$classe <- classe
testRemove <- grepl("^X|timestamp|window", names(testRaw))
testRaw <- testRaw[, !testRemove]
testCleaned <- testRaw[, sapply(testRaw, is.numeric)]

inTrain <- createDataPartition(trainCleaned$classe, p = 0.70, list = FALSE)

trainData <- trainCleaned[inTrain, ]
testData <- trainCleaned[-inTrain, ]

controlRf <- trainControl(method="cv", 5)
modFit <- train(classe ~ ., data=trainData, method="rf", trControl=controlRf, ntree=250)
modFit

predictRf <- predict(modelFit, testData)
cM <- confusionMatrix(testData$classe, predictRf)

accuracy <- postResample(predictRf, testData$classe)
result <- predict(modelRf, testCleaned[, -length(names(testCleaned))])
# Removing columns that 
# knit("project.Rmd")
# markdownToHTML("project.md", "project.html", options = c("use_xhml"))

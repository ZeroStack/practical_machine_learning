
Practical Machine Learning Course: Project
========================================================

### Introduction

Using devices such as Jawbone Up, Nike FuelBand, and Fitbit it is now possible to collect a large amount of data about personal activity relatively inexpensively. These type of devices are part of the quantified self movement – a group of enthusiasts who take measurements about themselves regularly to improve their health, to find patterns in their behavior, or because they are tech geeks. One thing that people regularly do is quantify how much of a particular activity they do, but they rarely quantify how well they do it. 

This document will attempt to demonstrate the use of data from accelerometers to predict the manner in which the exercise was conducted.

### Data

The training data for this project are available here:

https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv

The test data are available here:

https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv

The data for this project come from this source: http://groupware.les.inf.puc-rio.br/har. If you use the document you create for this class for any purpose please cite them as they have been very generous in allowing their data to be used for this kind of assignment.

### Load Relevant Libraries


```r
library(ggplot2)
library(caret)
```

### Download and Read Data

This section will download and read the data into R.


```r
# Download Data
trainUrl <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
testUrl <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"

trainFile <- "./data/pml-training.csv"
testFile  <- "./data/pml-testing.csv"

if (!file.exists("./data")) {
  dir.create("./data")
}
if (!file.exists(trainFile)) {
  download.file(trainUrl, destfile=trainFile, method="curl")
}
if (!file.exists(testFile)) {
  download.file(testUrl, destfile=testFile, method="curl")
}

# Read data
trainRaw <- read.csv("./data/pml-training.csv")
testRaw <- read.csv("./data/pml-testing.csv")
```

### Clean Data

In this step, we will clean the data. This will involve ridding missing values and meaningless variables that will deliver poorer results.

Check dimensions prior to any adjustments.


```r
# Check dimensions 
dim(trainRaw)
```

```
## [1] 19622   160
```

```r
dim(testRaw)
```

```
## [1]  20 160
```

Check for NA values


```r
sum(complete.cases(trainRaw))
```

```
## [1] 406
```

```r
sum(complete.cases(testRaw))
```

```
## [1] 0
```

Remove missing values


```r
trainRaw <- trainRaw[, colSums(is.na(trainRaw)) == 0] 
testRaw <- testRaw[, colSums(is.na(testRaw)) == 0] 
```

Remove columns that are not significant contributors.


```r
classe <- trainRaw$classe
trainRemove <- grepl("^X|timestamp|window", names(trainRaw))
trainRaw <- trainRaw[, !trainRemove]
trainCleaned <- trainRaw[, sapply(trainRaw, is.numeric)]
trainCleaned$classe <- classe
testRemove <- grepl("^X|timestamp|window", names(testRaw))
testRaw <- testRaw[, !testRemove]
testCleaned <- testRaw[, sapply(testRaw, is.numeric)]
```
Compare dimensions of trainRaw and trainCleaned

```r
dim(trainRaw) ; dim(trainCleaned)
```

```
## [1] 19622    87
```

```
## [1] 19622    53
```
### Create Data Partitions

Here, we will create data partitions. The training set will be formed of 70% of the trainRaw data after cleansing.


```r
set.seed(12345) # Reproducibile segment
inTrain <- createDataPartition(trainCleaned$classe, p = 0.70, list = FALSE)

trainData <- trainCleaned[inTrain, ]
testData <- trainCleaned[-inTrain, ]
```

### Data Modeling

I chose to fit a predictive model with the Random Forest algorithm. Random Forests grow many classification trees to classify a new object. This works on individual trees acting on a 'voting' system for that class.

The model is accurate at a trade-off for speed, which is not an immediate need.

For the Random Forest, I used a 5-fold cross validation.


```r
controlRf <- trainControl(method="cv", 5)
modelRf <- train(classe ~ ., data=trainData, method="rf", trControl=controlRf, ntree=250)
modelRf
```

```
## Random Forest 
## 
## 13737 samples
##    52 predictor
##     5 classes: 'A', 'B', 'C', 'D', 'E' 
## 
## No pre-processing
## Resampling: Cross-Validated (5 fold) 
## 
## Summary of sample sizes: 10989, 10990, 10990, 10989, 10990 
## 
## Resampling results across tuning parameters:
## 
##   mtry  Accuracy   Kappa      Accuracy SD  Kappa SD   
##    2    0.9913372  0.9890408  0.001042652  0.001319170
##   27    0.9903181  0.9877518  0.000983576  0.001244985
##   52    0.9857320  0.9819505  0.001190140  0.001505432
## 
## Accuracy was used to select the optimal model using  the largest value.
## The final value used for the model was mtry = 2.
```

### Apply Prediction Model to Partion Test Data Set

Apply prediction model to the test data set created at the data partition step.


```r
predictRf <- predict(modelRf, testData)
cM <- confusionMatrix(testData$classe, predictRf)
cM
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1673    1    0    0    0
##          B   13 1121    5    0    0
##          C    0   16 1007    3    0
##          D    0    0   24  940    0
##          E    0    0    0    3 1079
## 
## Overall Statistics
##                                           
##                Accuracy : 0.989           
##                  95% CI : (0.9859, 0.9915)
##     No Information Rate : 0.2865          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.986           
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9923   0.9851   0.9720   0.9937   1.0000
## Specificity            0.9998   0.9962   0.9961   0.9951   0.9994
## Pos Pred Value         0.9994   0.9842   0.9815   0.9751   0.9972
## Neg Pred Value         0.9969   0.9964   0.9940   0.9988   1.0000
## Prevalence             0.2865   0.1934   0.1760   0.1607   0.1833
## Detection Rate         0.2843   0.1905   0.1711   0.1597   0.1833
## Detection Prevalence   0.2845   0.1935   0.1743   0.1638   0.1839
## Balanced Accuracy      0.9960   0.9906   0.9840   0.9944   0.9997
```

Determine accuracy and out-of-sample error. 


```r
accuracy <- postResample(predictRf, testData$classe)
accuracy
```

```
##  Accuracy     Kappa 
## 0.9889550 0.9860251
```

The out-of-sample error is accuracy subtracted from 1.


```r
outSE <- 1 - as.numeric(cM$overall[1])
```

Our final values are:
Accuracy: 0.9940527 or 99.40%
Out-of-Sample Error: 0.0059473 or 0.60%

This is classed as a good predicton algorithm as it is abve the 80% accuracy threshold.

### Apply Prediction Model to Test Data Set

This is the final portion of the assignment; where we will apply the prediction algorithm to the 20 row test set downloaded in the beginning of the project.


```r
result <- predict(modelRf, testCleaned)
result
```

```
##  [1] B A B A A E D B A A B C B A E E A B B B
## Levels: A B C D E
```


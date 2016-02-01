# PredictionAssignment
Noah Hughes  
January 30, 2016  

[Github link](https://github.com/Noah-Hughes/MachineLearningProject)

#Introduction
In this project, your goal will be to use data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants. 

The goal of your project is to predict the manner in which they did the exercise. This is the "classe" variable in the training set. You may use any of the other variables to predict with. You should create a report describing how you built your model, how you used cross validation, what you think the expected out of sample error is, and why you made the choices you did. You will also use your prediction model to predict 20 different test cases.

##Loading the data

Check to see if files are download

```r
if(!file.exists("pml-training.csv")){
  download.file("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv","pml-training.csv")
}

if(!file.exists("pml-testing.csv")){
  download.file("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv","pml-testing.csv")
}
```

Check to see if files are download

```r
data = read.csv("pml-training.csv")
```

Remove the columns with Zero Variance

```r
library(caret)
```

```
## Loading required package: lattice
```

```
## Loading required package: ggplot2
```

```r
nearZeroColumns <- nearZeroVar(data)
data <- data[, -nearZeroColumns]
```

Get the columns that are missing more than 80% of the data

```r
sumMissing <- sapply(data, function(x) {sum((is.na(x) | x == ""))})

missingDataCol <- names(sumMissing[sumMissing > 0.8 * length(data$classe)])
```

Get the columns with text instead of data

```r
textCol <- c("X", "user_name", "raw_timestamp_part_1", "raw_timestamp_part_2","cvtd_timestamp", "new_window", "num_window")
```

Remove Columns no data and text data

```r
colRemove <- c(missingDataCol,textCol)
data <- data[, !names(data) %in% colRemove]
```

## Cross Validataion

We will split the data into a training and testing set in order to do cross validation of our model

```r
set.seed(1337)
inTrain <- createDataPartition(data$classe, p = 0.8, list = FALSE)
training <- data[inTrain, ]
testing <- data[-inTrain, ]
```

## Create Model

Using Random Forest we create our model

```r
library(randomForest)
```

```
## randomForest 4.6-12
```

```
## Type rfNews() to see new features/changes/bug fixes.
```

```
## 
## Attaching package: 'randomForest'
```

```
## The following object is masked from 'package:ggplot2':
## 
##     margin
```

```r
rf10Model <- randomForest(classe ~ ., data = training, importance = TRUE)
```


Check rf 10 tree models against prediction data

```r
rfPredict <- predict(rf10Model, testing)
rfmatix <- confusionMatrix(rfPredict, testing$classe)
print(rfmatix)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1116    3    0    0    0
##          B    0  756    5    0    0
##          C    0    0  678    4    2
##          D    0    0    1  637    1
##          E    0    0    0    2  718
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9954          
##                  95% CI : (0.9928, 0.9973)
##     No Information Rate : 0.2845          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9942          
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            1.0000   0.9960   0.9912   0.9907   0.9958
## Specificity            0.9989   0.9984   0.9981   0.9994   0.9994
## Pos Pred Value         0.9973   0.9934   0.9912   0.9969   0.9972
## Neg Pred Value         1.0000   0.9991   0.9981   0.9982   0.9991
## Prevalence             0.2845   0.1935   0.1744   0.1639   0.1838
## Detection Rate         0.2845   0.1927   0.1728   0.1624   0.1830
## Detection Prevalence   0.2852   0.1940   0.1744   0.1629   0.1835
## Balanced Accuracy      0.9995   0.9972   0.9947   0.9950   0.9976
```

The accuracy of our model is 99.5% which is pretty good.  Using cross validation the out of sample error rate is 0.5% (1-99.5%).  Looked at other models such as more Random Forest with number of trees at (10,30,100)  and also GBM.  This model was quick to run and had a low sample error rate. 

## Running against the Test Set

Running the model against the original course test set

```r
test = read.csv("pml-testing.csv")
testPredict <- predict(rf10Model, test)
testPredict
```

```
##  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 
##  B  A  B  A  A  E  D  B  A  A  B  C  B  A  E  E  A  B  B  B 
## Levels: A B C D E
```



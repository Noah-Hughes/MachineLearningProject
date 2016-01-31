library(caret)
training = read.csv("pml-training.csv")
testing = read.csv("pml-testing.csv")

# exclude near zero variance features
nzvcol <- nearZeroVar(training)
training <- training[, -nzvcol]

# exclude columns with m40% ore more missing values exclude descriptive
# columns like name etc
cntlength <- sapply(training, function(x) {
  sum(!(is.na(x) | x == ""))
})

nullcol <- names(cntlength[cntlength < 0.6 * length(training$classe)])
descriptcol <- c("X", "user_name", "raw_timestamp_part_1", "raw_timestamp_part_2", 
                 "cvtd_timestamp", "new_window", "num_window")
excludecols <- c(descriptcol, nullcol)

training <- training[, !names(training) %in% excludecols]

library(randomForest)

system.time(rfModel <- randomForest(classe ~ ., data = training, importance = TRUE, ntrees = 10))

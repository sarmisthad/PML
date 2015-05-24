library("caret");

## ==================== Step 1==================================
# Reading training data (pml-training)
# replacing "#DIV/0!" values by "NA"
training_raw <- read.csv("Data/pml-training.csv", header=TRUE, sep=",",na.strings=c("NA","#DIV/0!"));
# summary(training_raw);


## ==================== Step 2==================================
# Data pre processing

# Setting all NA fields by 0 and adding two dummy parameters for new_window "yes"/"no" , instead of new_window(6th column)
# Removing redundant field cvtd_timestamp, as timestamp already captured in raw_timestamp_part_1 and raw_timestamp_part_2 as numeric data
training_raw[is.na(training_raw)] <- 0;
dummies <- dummyVars(~ new_window, data=training_raw);
training <- cbind(training_raw[,-c(5,6)],predict(dummies, newdata=training_raw));

# Checking variability of the parameters and removing parameters with zero variability, as well as first two columns of row index and names
nsv<- nearZeroVar(training[,c(3:160)],saveMetrics=TRUE);
training_cleandata <- training[,-c(1,2,12,15,24,87,90,99,125,128,137)];

# Principal component analysis with variability threshold=0.9
r = preProcess(training_cleandata[,-147], method = "pca", thresh = 0.9);
train_r <- predict(r,training_cleandata[,-147]);

modelfit_PCA = train(training_cleandata[,147] ~ ., method = 'gbm', preProcess="pca", data = training_cleandata[,-147], verbose=FALSE);

ctrl <- trainControl(method="repeatedcv",repeats = 3); #,classProbs=TRUE,summaryFunction = twoClassSummary) 
knnFit <- train(training_cleandata[,147] ~ ., data = training_cleandata[,-147], method = "knn", trControl = ctrl, preProcess = r, tuneLength = 20, verbose=FALSE) 


# ===== Step 3 : read and clean test data and apply model to predict
testing_raw <- read.csv("Data/pml-testing.csv", header=TRUE, sep=",",na.strings=c("NA","#DIV/0!"));
testing_raw[is.na(training_raw)] <- 0;
dummies_t <- dummyVars(~ new_window, data=testing_raw);
testing <- cbind(testing_raw[,-c(5,6)],predict(dummies_t, newdata=testing_raw));
testing_cleandata <- testing[,-c(1,2,12,15,24,87,90,99,125,128,137)];

predictions<-predict(knnFit, newdata=testing_cleandata);


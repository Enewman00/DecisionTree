#Ethan Newman
#EEN170000
#CS 4375.0W2
library(mlbench)
library(rpart)
library(rpart.plot)
library(ggplot2)
library(lattice)
library(e1071)
library(caret)
library(parallel)
library(iterators)
library(foreach)
library(doParallel)

?rpart
#1-Improve the quality of “bank” dataset by handling duplicate objects and missing
#values. Using the default setting of rpart, create a decision tree to classify the improved bank objects into two categories: 
#desired target (Yes) and not desired target (No). The client is a desired target, if the client has subscribed a term deposit.

#read in bank-additional.csv
bank <- read.csv("~/bank-additional.csv", sep = ";")
#count rows and columns
nrow(bank)
ncol(bank)
#remove duplicate rows
bank <- unique(bank)
#remove objects with missing values
bank <- na.omit(bank)

#use rpart to create decision tree
tree <- rpart(y~., data = bank)
#A. Plot the tree
rpart.plot(tree, extra = 2, under = TRUE, varlen=0, faclen=0)
#B. Describe the characteristics of the objects that were classified as desired target.
tree
#The characteristics of the objects that were classified as desired target are:
#They are nr.employee >= 5088, and duration >= 616.5 and duration >= 837.5 and cons.conf.idx >= -42.35 and job is admin, blue-collar, entrepreneur, housemaid, retired, or technician
#They are nr.employee >= 5088, and duration >= 616.5 and duration >= 837.5 and cons.conf.idx < -42.35
#They are nr.employee < 5087, and duration >= 165.5 and pdays < 13.5
#They are nr.employee < 5087, and duration >= 165.5 and pdays >= 13.5 and duration >= 390.5
#They are nr.employee < 5087, and duration >= 165.5 and pdays >= 13.5 and duration < 390.5 and contact = cellular and month = dec, jun, may
#They are nr.employee < 5087, and duration >= 165.5 and pdays >= 13.5 and duration < 390.5 and contact = cellular and month = apr, aug, jul, mar, nov, oct, sep and job=housemaid, self-employed, unemployed

#C. Predict a class for each object in the improved bank data set using the constructed tree. Find TPR, FPR, TNR, FNR, and accuracy for this prediction.
pred <- predict(tree, bank, type="class")
pred
confusion <- table(bank$y, pred)
confusion
correct <- sum(diag(confusion))
correct
error <- sum(confusion) - correct
error
error / nrow(bank)
accuracy <- correct / (correct + error)
accuracy
#[TN FP]
#[FN TP]
tpr <- confusion[2,2] / (confusion[2,2] + confusion[2,1])
tnr <- confusion[1,1] / (confusion[1,1] + confusion[1,2])
fpr <- confusion[1,2] / (confusion[1,1] + confusion[1,2])
fnr <- confusion[2,1] / (confusion[2,2] + confusion[2,1])
tpr
tnr
fpr
fnr

#D. Display the actual class label and the predicted class label for the first ten objects of the improved bank data set. Find the error rate for these ten objects.
pred10 <- predict(tree, bank[1:10,], type="class")
pred10
#actual
bank[1:10,21]
confusion10 <- table(bank[1:10,]$y, pred10)
confusion10
correct10 <- sum(diag(confusion10))
correct10
error10 <- sum(confusion10) - correct10
error10
#E. After finding the training error, estimate generalization error for this tree using pessimistic and optimistic approach. 
#optimistic: errors on testing set = error on training set
error / nrow(bank)
#pessimistic: errors on testing set = (e(t) + N * 0.5
(error) + (12 * 0.5)
((error) + (12 * 0.5)) / nrow(bank)


#2-Using rpart, create a fully grown decision tree to classify the objects in the improved bank into two categories: desired target (Yes) and not desired target (No).
#We can control splitting the node using minsplit and cp. To have a fully grown decision tree, consider minsplit as 2 and cp as 0 for this tree.
tree_full <- rpart(y ~., data=bank, control=rpart.control(minsplit=2, cp=0))
#rpart.plot(tree_full, extra = 2, under = TRUE,  varlen=0, faclen=0)


#A. Predict a class for each object in the improved bank data set using the constructed tree. Find TPR, FPR, TNR, FNR, and accuracy of this prediction.
pred_full <- predict(tree_full, bank, type="class")
pred_full
confusion_full <- table(bank$y, pred_full)
confusion_full
correct_full <- sum(diag(confusion_full))
correct_full
error_full <- sum(confusion_full) - correct_full
error_full
error_full / nrow(bank)
accuracy_full <- correct_full / (correct_full + error_full)
accuracy_full
#[TN FP]
#[FN TP]
tpr_full <- confusion_full[2,2] / (confusion_full[2,2] + confusion_full[2,1])
tnr_full <- confusion_full[1,1] / (confusion_full[1,1] + confusion_full[1,2])
fpr_full <- confusion_full[1,2] / (confusion_full[1,1] + confusion_full[1,2])
fnr_full <- confusion_full[2,1] / (confusion_full[2,2] + confusion_full[2,1])
tpr_full
tnr_full
fpr_full
fnr_full
#B. After finding the training error, estimate generalization error for this tree using pessimistic and optimistic approach.
#optimistic: testing error = training error
error_full
#pessimistic: 
#pessimistic: errors on testing set = (e(t) + N * 0.5
(error) + (sum(tree_full$frame$var == "<leaf>") * 0.5)
((error) + (sum(tree_full$frame$var == "<leaf>") * 0.5)) / nrow(bank)

#C. Compare the training error based on the fully grown tree (constructed tree in Q2) and the default tree (constructed tree in Q1). Which one has a higher
#training error rate? Why? 
#The fully grown tree has a much lower training error than the default tree. This is because the fully grown tree has an error of 0, due to it being fully fit for the
#training data, meaning it will have 0 errors


#3-What is under-fitting in machine learning? Construct a decision tree for the improved bank data set to illustrate under-fitting. 
#Model underfitting is when the model is too simple. it causes the training error and the test error to be too large
tree_under <- rpart(y ~ ., data=bank, control=rpart.control(maxdepth = 3))
rpart.plot(tree_under, extra = 2, under = TRUE,  varlen=0, faclen=0)


#4-Randomly select 2/3 of the objects in the improved bank data set as the training dataset and 1/3 of the objects as the test dataset. 
#Create a decision tree using the training set. Then, randomly select 150 objects of the improved bank data
#set as the training dataset and consider the remaining objects as the test dataset.
#Create another decision tree using this training set. Find the training error and
#testing error for both trees. Which one has the higher the training error and testing error? Why? 

#create copy of bank
#get size of training split
n_train <- as.integer(nrow(bank)*.66)
n_train
#random sample training indexes
train_id <- sample(1:nrow(bank), n_train)
#split
train <- bank[train_id,]
test <- bank[-train_id, colnames(bank) != "y"]
test_type <- bank[-train_id, "y"]

# create a tree using the training set (2/3 of the zoo data set )
tree_third <- rpart(y ~., data=train,control=rpart.control(minsplit=2))
rpart.plot(tree_third, extra = 2, under = TRUE,  varlen=0, faclen=0)

#150 - training, rest = testing
train150_id <- sample(1:nrow(bank), 150)
train150 <- bank[train150_id,]
#split
test150 <- bank[ -train150_id, colnames(bank) != "y"]
test150_type <- bank[-train150_id, "y"]

# create a tree using the training set (150 of the zoo data set )
tree_150 <- rpart(y ~., data=train150,control=rpart.control(minsplit=2))
rpart.plot(tree_150, extra = 2, under = TRUE,  varlen=0, faclen=0)

#test accuracy
accuracy <- function(truth, prediction) {
  tbl <- table(truth, prediction)
  sum(diag(tbl))/sum(tbl)
}
#test 2/3 and 1/3
accuracy(train$y, predict(tree_third, train, type="class"))
accuracy(test_type, predict(tree_third, test, type="class"))

#test 150 and rest
accuracy(train150$y, predict(tree_150, train150, type="class"))
accuracy(test150_type, predict(tree_150, test150, type="class"))

#The tree that uses 2/3 of the data for the training set has the higher training error but the lower testing error
#The tree that uses 150 items of the data for the training set has the lower training error but a higher testing error
#This is because the tree that uses 150 items of the data for the training set has the tree specifically fitted for that training set
#leaving no training error, but the data isn't diverse enough to provide a good tree for testing data to fit well into.



#5-After shuffling the improved bank data set, Partition the dataset into 10 disjoint
#subsets. Consider the first partition (fold) as the testing set and the remaining
#partitions (folds) as the training set. Find the size of the training set and test set.
#Using rpart, create a decision tree based on the training set. (consider minimum 
#split as 2). Calculate the training error and the testing error . Repeat this process
#for every other folds. Find the average of testing error. 

#shuffle the improved bank data set
index <- 1:nrow(bank)
index <- sample(index) ### shuffle index

#partition dataset into 10 disjoint subsets
fold <- rep(1:10, each=nrow(bank)/10)[1:nrow(bank)]
fold

folds <- split(index, fold) ### create list with indices for each fold
folds



trainingAccs <- vector(mode="numeric")
testingAccs <- vector(mode="numeric")
trainSizes <- vector(mode="numeric")
testSizes <- vector(mode="numeric")
for(i in 1:length(folds)) {
  #find size of training set 
  trainSizes[i] <- length(folds[[i]])
  #find size of test set
  testSizes[i] = nrow(bank) - length(folds[[i]])
  
  #create decision tree (min split = 2)
  tree_loop <- rpart(y ~., data=bank[-folds[[i]],], control=rpart.control(minsplit=2))
  #training accuracy
  trainingAccs[i] <- accuracy(bank[folds[[i]],]$y, predict(tree_loop, bank[folds[[i]],], type="class"))
  #testing accuracy
  #try - for every index EXCEPT
  foldsType <- bank[-folds[[i]], "y"]
  testingAccs[i] <- accuracy(foldsType, predict(tree_loop, bank[-folds[[i]],], type="class"))
  
}
trainSizes
testSizes
trainingAccs
testingAccs
mean(testingAccs)

#calculate training error and testing error
#repeat for every other folds.
#find average of testing error









# Load the Dataset
advertising <- read.csv(file.choose()) # Choose the claimants Data set

sum(is.na(advertising))

# Omitting NA values from the Data 
advertising1 <- na.omit(advertising) # na.omit => will omit the rows which has atleast 1 NA value
dim(advertising1)

###########

colnames(advertising)
advertising <- advertising[ , -(5:9)] # Removing the first column which is is an Index

# Preparing a linear regression 
mod_lm <- lm(Clicked_on_Ad ~ ., data = advertising)
summary(mod_lm)

pred1 <- predict(mod_lm, advertising)
pred1
# plot(claimants$CLMINSUR, pred1)

# We can also include NA values but where ever it finds NA value
# probability values obtained using the glm will also be NA 
# So they can be either filled using imputation technique or
# exlclude those values 


# GLM function use sigmoid curve to produce desirable results 
# The output of sigmoid function lies in between 0-1
model <- glm(Clicked_on_Ad ~ ., data = advertising, family = "binomial")
summary(model)
# We are going to use NULL and Residual Deviance to compare the between different models

# To calculate the odds ratio manually we going r going to take exp of coef(model)
exp(coef(model))

# Prediction to check model validation
prob <- predict(model, advertising, type = "response")
prob

# or use plogis for prediction of probabilities
prob <- plogis(predict(model, advertising))

# Confusion matrix and considering the threshold value as 0.5 
confusion <- table(prob > 0.5, advertising$Clicked_on_Ad)
confusion

# Model Accuracy 
Acc <- sum(diag(confusion)/sum(confusion))
Acc

# Convert the probabilities to binary output form using cutoff
pred_values <- ifelse(prob > 0.5, 1, 0)

library(caret)
# Confusion Matrix
confusionMatrix(factor(advertising$Clicked_on_Ad, levels = c(0, 1)), factor(pred_values, levels = c(0, 1)))


# Build Model on 100% of data
advertising1 <- advertising1[ ,-(5:9)] # Removing the first column which is is an Index
# To Find the optimal Cutoff value:
# The output of sigmoid function lies in between 0-1
fullmodel <- glm(Clicked_on_Ad ~ ., data = advertising1, family = "binomial")
summary(fullmodel)

prob_full <- predict(fullmodel, advertising1, type = "response")
prob_full

# Decide on optimal prediction probability cutoff for the model
library(InformationValue)
optCutOff <- optimalCutoff(advertising1$Clicked_on_Ad, prob_full)
optCutOff

# Check multicollinearity in the model
library(car)
vif(fullmodel)

# Misclassification Error - the percentage mismatch of predcited vs actuals
# Lower the misclassification error, better the model.
misClassError(advertising1$Clicked_on_Ad, prob_full, threshold = optCutOff)


# ROC curve
# Greater the area under the ROC curve, better the predictive ability of the model
plotROC(advertising1$Clicked_on_Ad, prob_full)


# Confusion Matrix
predvalues <- ifelse(prob_full > optCutOff, 1, 0)

results <- confusionMatrix(predvalues, advertising1$Clicked_on_Ad)

sensitivity(predvalues, advertising1$Clicked_on_Ad)
confusionMatrix(actuals = advertising1$Clicked_on_Ad, predictedScores = predvalues)


###################
# Data Partitioning
n <- nrow(advertising1)
n1 <- n * 0.85
n2 <- n - n1
train_index <- sample(1:n, n1)
train <- advertising1[train_index, ]
test <- advertising1[-train_index, ]

# Train the model using Training data
finalmodel <- glm(Clicked_on_Ad ~ ., data = train, family = "binomial")
summary(finalmodel)

# Prediction on test data
prob_test <- predict(finalmodel, newdata = test, type = "response")
prob_test

# Confusion matrix 
confusion <- table(prob_test > optCutOff, test$Clicked_on_Ad)
confusion

# Model Accuracy 
Accuracy <- sum(diag(confusion)/sum(confusion))
Accuracy 

# Creating empty vectors to store predicted classes based on threshold value
pred_values <- NULL
pred_values <- ifelse(prob_test > optCutOff, 1, 0)

# Creating new column to store the above values
test[,"prob"] <- prob_test
test[,"pred_values"] <- pred_values

table(test$Clicked_on_Ad, test$pred_values)


# Compare the model performance on Train data
# Prediction on test data
prob_train <- predict(finalmodel, newdata = train, type = "response")
prob_train

# Confusion matrix
confusion_train <- table(prob_train > optCutOff, train$Clicked_on_Ad)
confusion_train

# Model Accuracy 
Acc_train <- sum(diag(confusion_train)/sum(confusion_train))
Acc_train


# Statistical Analysis Happiness Score 2019-R
#############################################################################
# Multiple Linear Regression

happy <- read.csv('WorldHappinessReport_2019.csv', header = TRUE) # read file

df <- as.data.frame(happy[,c('Country.or.region', 'Score', 'GDP.per.capita', 
                             'Social.support', 'Healthy.life.expectancy', 'Freedom.to.make.life.choices', 
                             'Generosity','Perceptions.of.corruption')])

happy <- as.data.frame(happy[,c( 'Score', 'GDP.per.capita', 'Social.support', 'Healthy.life.expectancy', 
                                 'Freedom.to.make.life.choices', 'Generosity', 'Perceptions.of.corruption')])

# First, we fit a multiple linear regression model. Score is the dependent variable.

fit <- lm(Score ~ GDP.per.capita + Social.support + Healthy.life.expectancy + Freedom.to.make.life.choices + 
            Generosity + Perceptions.of.corruption, data=happy)

summ_fit <- summary(fit)

summary(fit)

# Since the p-value of Generosity and Perceptions.of.corruption are more than 0.05, they are not significant. 
# Also,  R2 , coefficient of determination, is a measure of goodness of fitness of the model, 
# between 0 and 1. In this case,  R2  equals 0.779, which is relatively a good fit.

# try a fit with excluding the above two insignificant predictors
fit2 <- lm(Score ~ GDP.per.capita + Social.support + Healthy.life.expectancy + Freedom.to.make.life.choices
           , data=happy)

summary(fit2)

#############################################################################
# Model selection
# Best subset selection method

# load library leaps
library(leaps)

# run regsubsets using Score as the response, with up to 6 predictors
# store the results in a variable called regfit.full
regfit.full = regsubsets(Score~., data = happy,nvmax = 6)
summary(regfit.full)
reg.summary = summary(regfit.full)

# plot adjusted R^2 as a function of predictors in the model
plot(reg.summary$adjr2, xlab="Number of predictors", ylab="Adjusted R^2")

# display the number of predictors for which adjusted R^2 reaches its maximum
which.max(reg.summary$adjr2)

# run regsubsets using Score as the response, with up to 5 predictors
# store the results in a variable called regfit.full
regfit.full = regsubsets(Score~., data = happy,nvmax = 5)
summary(regfit.full)
reg.summary = summary(regfit.full)

# plot Cp as a function of predictors in the model
plot(reg.summary$cp, xlab="Number of predictors", ylab="Cp")

# display the number of predictors for which Cp reaches its minimum
which.min(reg.summary$cp)

# the coefficient estimates for the best model using best subset selection method
# we selected 4 predictors for more efficient
coef(regfit.full,4)

######################################
# Forward stepwise selection method
# run forward stepwise selection, allowing subsets with up to 6 predictors
regfit.fwd=regsubsets(Score~.,data=happy,nvmax=6, method="forward")
summary(regfit.fwd)
reg.summary = summary(regfit.fwd)

# plot Cp as a function of predictors in the model
plot(reg.summary$cp, xlab="Number of predictors", ylab="Cp")

# display the number of predictors for which Cp reaches its minimum
which.min(reg.summary$cp)

# the coefficient estimates for the best model using forward selection method
# Again, we also selected 4 predictors for more efficient
coef(regfit.fwd,4)

#############################################################################

# Shrinkage methods LASSO

library(glmnet)
# the data frame x will hold the data for predictors
x=model.matrix(Score~.,data=happy)[,-1] 

# the vector y will hold the data for the response, Score in this case
y=happy$Score

# Use 5-fold cross-validation on the whole data, to determine the best lambda value
set.seed(1)
cv.out=cv.glmnet(x,y,nfolds=5,alpha=1)

# plot the MSE observed in the cross-validation as a function of log(lambda)
plot(cv.out)

# determine which lambda minimized the MSE, call it bestlam, and display it
bestlam=cv.out$lambda.min
bestlam

# Here, we can select 5 predictors for more efficient, corresponding to log(lambda) ~ -2.5
# display coefficients
lasso.final=glmnet(x, y, alpha=1, lambda=exp(-2.5))
coef(lasso.final)


#############################################################################
# Classification method
# K-Nearest Neighbors

library(class)
set.seed(1)

# Countries with a score higher than the average Happiness Score are in group 'High', 
# and lower or equal to the mean value are in group 'Low'.

happy$Score.index <- factor(happy$Score> mean(happy$Score), levels=c(TRUE,FALSE),labels=c("High", "Low"))


# split data into two subsets
train=sample(1:nrow(happy), nrow(happy)/2) 

# next, we create the part of x and y that will be our training data
# we will call these x.train and y.train
happy.train=happy[train,2:7]
happy.test=happy[-train,2:7]

HighScore.train=happy$Score.index[train]
HighScore.test=happy$Score.index[-train]

# run knn with K=1 and store prediction results in a variable called knn.pred
knn.pred = knn(happy.train,happy.test,HighScore.train,k=1)

# display a table that shows prediction results
table(knn.pred,HighScore.test)
round(prop.table(table(knn.pred,HighScore.test)),2)

# calculate prediction accuracy
mean(knn.pred==HighScore.test)

# run knn with K=3 and store prediction results in a variable called knn.pred
knn.pred = knn(happy.train,happy.test,HighScore.train,k=3)

# display a table that shows prediction results
table(knn.pred,HighScore.test)
round(prop.table(table(knn.pred,HighScore.test)),2)

# calculate prediction accuracy
mean(knn.pred==HighScore.test)

# run knn with K=5 and store prediction results in a variable called knn.pred
knn.pred = knn(happy.train,happy.test,HighScore.train,k=5)

# display a table that shows prediction results
table(knn.pred,HighScore.test)
round(prop.table(table(knn.pred,HighScore.test)),2)

# calculate prediction accuracy
mean(knn.pred==HighScore.test)

# Conclusion: K = 3 works best to predict the Happiness

#############################################################################

# Decision Trees
# Create a decision tree that predicts whether the Score.index for a country will be high or low than the mean
library(tree)
set.seed(1)
happy$Score.index <- factor(happy$Score> mean(happy$Score), levels=c(TRUE,FALSE),labels=c("High", "Low"))
happy1=happy[,-1] 

# nrow(happy1) will give us the number of rows in happy1. we are randomly picking 
# half of those rows to be our training data. We are storing the row numbers in our training data 
# in a variable called train. 
train = sample(1:nrow(happy1), nrow(happy1)/2)

# the remaining rows are set aside as test data
happy.test = happy1[-train,]

# we store the actual High vs. Low observations for test data in a vector called High.test 
Score.index.test = happy1$Score.index[-train]

# create a decision tree to predict high or low score index, using only training data. 
# The response is Score.index, the predictors are everything except Score.index
tree.happy = tree(Score.index ~ . , happy1[train,])
summary(tree.happy)

# plot the tree
plot(tree.happy)

# see what the branches are 
text(tree.happy,pretty=0)

# use cross-validation and pruning to obtain smaller trees
cv.happy = cv.tree(tree.happy,FUN=prune.misclass)

# check size and dev
names(cv.happy)

# display the information in cv.happy
# observe that dev is minimum when size is 6
cv.happy

# use a function called prune.misclass() to obtain the best tree
prune.happy = prune.misclass(tree.happy,best=6)

# plot the best tree
plot(prune.happy)

# display branch names on the tree
text(prune.happy, pretty = 0)

# using the best tree obtained we make High vs. Low predictions for the test data
# the predictions are stored in a vector called tree.pred
tree.pred = predict(prune.happy,happy.test,type="class")

# display a table that shows predictions for test data versus actuals for test data
table(tree.pred, Score.index.test)
round(prop.table(table(tree.pred,Score.index.test)),2)

# calculate prediction accuracy
mean(tree.pred==Score.index.test)

#############################################################################


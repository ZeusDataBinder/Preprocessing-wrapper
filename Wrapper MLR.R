install.packages("mlr")
install.packages("gbm")
library(gbm)
library(mlbench)
library(mlr)

data(iris)
str(iris)


## Define the task
taskC = makeClassifTask(data = iris,target = "Species")
str(taskC)

getTaskFormula(taskC) #Generated the target variable

getTaskType(taskC)#type of task

getTaskId(taskC)#name of the dataset, showed as id, id is used for plotting / comparision with other id's

getTaskTargetNames(taskC)#Name of the target variable

getTaskSize(taskC) #No of rows in the dataset

getTaskNFeats(taskC)#No. of features

str(getTaskData(taskC))

getTaskFeatureNames(taskC)#Generate feature names

head(getTaskTargets(taskC))
tail(getTaskTargets(taskC))

#Normalizing features
taskC = normalizeFeatures(taskC, method = "range")
summary(getTaskData(taskC))



## Define the learner

learner = makeLearner("classif.lda")#Here the learning algorithm is Linear Discriminant Analysis

#Accessing the core components of the learning module
learner$id
learner$type
learner$package
learner$properties
learner$name
learner$predict.type
learner$par.vals
learner$par.set#accessing the hyperparametrs of the learning model created above

getHyperPars(learner)#Current hyperparametrs

getParamSet(learner)#same as learner$par.set

#Knowing info of other alogorithms used for learning
getParamSet("classif.randomForest")

getParamSet("regr.gbm")#Regression Learning using Gradient Boosting machines 

learner=setLearnerId(learner,"HelloWorld")#Changed the id of the learning module
learner

learner = setPredictType(learner, "prob")#changed the predict type from respnse to probability
learner

### Train the learner with the task generated above
#Here the entire dataset is used for training
Iristrainer = train(learner, taskC)
Iristrainer

#Accessing the IRIS trained learning model 
names(Iristrainer)
Iristrainer$learner
Iristrainer$features
Iristrainer$time
getLearnerModel(Iristrainer)

#Size of the data in task
n = getTaskSize(taskC)

#Here 1/3 of the dataset is used for training.
train.set = sample(n, size = n/3)

#Applying the 1/3 data to train again
Iristrainer = train("classif.lda", taskC, subset = train.set)
Iristrainer


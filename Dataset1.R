# Importing the dataset
dataset = read.csv('dataset1.csv')

# Encoding categorical data
dataset$Swimming.Pool = factor(dataset$Swimming.Pool,
                         levels = c('YES','NO'),
                         labels = c(1, 0))
dataset$Exercise.Room = factor(dataset$Exercise.Room,
                           levels = c('YES', 'NO'),
                           labels = c(1, 0))
dataset$Free.Wifi = factor(dataset$Free.Wifi,
                               levels = c('YES', 'NO'),
                               labels = c(1, 0))
dataset$Basketball.Court = factor(dataset$Basketball.Court,
                               levels = c('YES', 'NO'),
                               labels = c(1, 0))
dataset$Yoga.Classes = factor(dataset$Yoga.Classes,
                               levels = c('YES', 'NO'),
                               labels = c(1, 0))
dataset$Club = factor(dataset$Club,
                               levels = c('YES', 'NO'),
                               labels = c(1, 0))
dataset$Traveler.type = factor(dataset$Traveler.type,
                               levels = c('Friends', 'Families','Solo','Couples','Business'),
                               labels = c(1, 2,3,4,5))

library(dummies)
dataset_new = dummy.data.frame(dataset,sep=".")

#removing negative value from member years
dataset_new$Member.years = ifelse(dataset_new$Member.years < 0, 0,dataset_new$Member.years )

# ensure the results are repeatable
set.seed(7)
# load the library
library(mlbench)
library(caret)
# calculate correlation matrix
correlationMatrix <- cor(dataset_new)


#Remove unwanted variables
dataset = dataset[,-1]
dataset = dataset[,-13]
dataset = dataset[,-15]
dataset = dataset[,-5]
dataset = dataset[,-13]
dataset = dataset[,-12]
dataset = dataset[,-14]
dataset = dataset[,-13]

#removing negative value from member years
dataset$Member.years = ifelse(dataset$Member.years < 0, 0,dataset$Member.years )


#distribute dataset into training and test set
library(caTools)
set.seed(123)
split = sample.split(dataset$Score, SplitRatio = 0.7)
training_set = subset(dataset, split == TRUE)
test_set = subset(dataset, split == FALSE)

training_set[,1:3]= scale(training_set[,1:3])
training_set[,12]= scale(training_set[,12])
test_set[,1:3]= scale(test_set[,1:3])
test_set[,12]= scale(test_set[,12])


#fit LDA
library(MASS)
lda = lda(formula = Score ~ ., data = training_set)
training_set = as.data.frame(predict(lda, training_set))
training_set = training_set[c(7, 8, 9, 10, 1)]
test_set = as.data.frame(predict(lda, test_set))
test_set = test_set[c(7, 8,9,10, 1)]


#fit SVM
library(e1071)
classifier = svm(formula = class ~ .,
                 data = training_set,
                 type = 'C-classification',
                 kernel = 'linear')

#predict test set score
y_pred = predict(classifier, newdata = test_set[-5])

#build confusion matrix
cm = table(test_set[, 5], y_pred)

#print confusion matrix
print(cm, mode = cm$mode, digits = max(3,getOption("digits") - 3), printStats = TRUE)

#Result accuracy
accuracy<-sum(diag(cm))/sum(cm)

#print accuracy
print(accuracy*100)


#To find out most relevant features in model

# ensure the results are repeatable
set.seed(7)

# load the library
install.packages('mlbench')
library(mlbench)
library(caret)

# define the control using a random forest selection function
control <- rfeControl(functions=rfFuncs, method="cv", number=10)

# run the RFE algorithm
results <- rfe(dataset[,1:11], dataset[,12], sizes=c(1:11), rfeControl=control)

# summarize the results
print(results)

# plot the results
plot(results, type=c("g", "o"))
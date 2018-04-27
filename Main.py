__author__ = "Kashmira Dolas"
"""
language:   Python3
file:   DataPreperation.py
author: kashmira Dolas
description: Load data, train and test using Logistic regression, Decision
Tree and Support Vector Machine
attributes to numeric
"""

import csv
from sklearn.cross_validation import train_test_split
from sklearn import metrics
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
import time


def readandconvert(file):
    """
    Read the file and convert it to the required format.

    :Pre-Req: This function has been written to the read the bank.csv data 
    set. Modification may be required to read any other csv file.

    :param file: Name of the file to read
    :return: dataset and target vector
    """
    dataset = []
    with open(file) as csvfile:
        readCSV = csv.reader(csvfile, delimiter=',')
        next(readCSV, None)
        for row in readCSV:
            temp = []
            for value in row:
                temp.append(float(value))
            dataset.append(temp)

    data = []
    target = []
    for i in dataset[1:]:
        data.append(i[:16])
        target.append(i[16])

    return data, target


def train(model, data, target):
    """
    Split the data into training and testing set.
    Train the model on the training set and predict the values for the testing 
    set.
    
    :param model: Model to be trained
    :param data: Data to be split
    :param target: target vector
    :return: testing target vecto, predicted values vector, time required to 
    build the model
    """
    training_set, testing_set, training_target, testing_target = \
        train_test_split(data, target, test_size=0.3,
                         random_state=1)
    starttime = int(time.time()*1000)
    model = model.fit(training_set, training_target)
    endtime = int(time.time()*1000)
    predict = model.predict(testing_set)
    return model, testing_target, predict, endtime-starttime


def kcrossvalidation_score(model, data, target, k = 10):
    """
    Calculate the k fold cross validation scores
    :param model: The model to be trained
    :param data: Training data
    :param target: Target vector
    :param k: number of folds to be calculated
    :return: Cross validation scores
    """

    scores = cross_val_score(model, data, target, cv=k)

    print(str(k)+"-Fold Cross Validation Scores: ", scores)
    return scores


def main():
    """
    Main function.
    
    Pre-requisites: TO run this code you need to sklearn, numpy, scipy installed
    :return: 
    """

    data, target = readandconvert(input("Please enter the name of the input "
                                        "file without .csv ")+".csv")

    # Training a Logistic Regression model
    print("LOGISTIC REGRESSION")
    scores = kcrossvalidation_score(LogisticRegression(random_state=2), data,
                               target)
    model_LR, testing_target, predict, timetaken = train(LogisticRegression(),
                                                      data,
                                              target)
    print("Time Taken to build the model = " + str(timetaken) + "nsecs")
    print("Mean Squared Error = " + str(metrics.mean_squared_error(
        testing_target, predict)))
    print("Accuracy = " + str(metrics.accuracy_score(testing_target, predict)
                              * 100) + " %\n\n")

    # Training a Linear SVM model
    print("Linear SVM")
    kcrossvalidation_score(SVC(kernel='rbf', C=0.5), data, target)
    model_SVM, testing_target, predict, timetaken = train(SVC(kernel='rbf', 
    C=0.5),data,target)
    print("Time Taken to build the model = "+str(timetaken)+"nsecs")
    print("Accuracy = " + str(metrics.accuracy_score(testing_target, predict)
                              * 100) + " %\n\n")

    # Training a Decision Tree
    print("DECISION TREE")
    kcrossvalidation_score(DecisionTreeClassifier(random_state=10, 
      min_samples_split=500, criterion='entropy', max_depth=5),data,target)

    model_DT, testing_target, predict, timetaken = train(DecisionTreeClassifier(random_state=10, 
                                                  min_samples_split=500, 
                                                  criterion='entropy', 
                                                  max_depth=5),data,target)
    print("Time Taken to build the model = " + str(timetaken) + "nsecs")
    print("Accuracy = " + str(metrics.accuracy_score(testing_target, predict)
     * 100)+" %\n\n")


if __name__ == '__main__':
    main()
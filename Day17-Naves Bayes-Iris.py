#NAIVE's BAYES for IRIS dataset

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

path=("D:/CDAC-AI/Training Material/Machine Learning/Class Codes/iris.data")
headernames = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'Class']
print(headernames)
data=pd.read_csv(path, names=headernames)
pd.set_option('max_columns', None)
print(data)

#Defining X and Y
inputs=data.drop('Class',axis='columns')# x-axis
target=data.Class # y-axis
print('X dataset:',inputs.head(10))
print('Y dataset:',target.head(10))

#Splitting data in train and test set
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(inputs,target,test_size=0.3)

from sklearn.naive_bayes import GaussianNB
model=GaussianNB()
model.fit(X_train,y_train)
print(model.score(X_train,y_train))
print('X_test dataset:',X_test[0:10])
print('Y_test dataset:',y_test[0:10])

print("predict X_test:")
print(model.predict(X_test[0:10]))

print("Probabilities of predicted X_test dataset:")
print(model.predict_proba(X_test[:10]))

#Calculating the score using cross validation
from sklearn.model_selection import cross_val_score
cvs=cross_val_score(GaussianNB(),X_train,y_train,cv=5)#Cv=cross validation=divide dataset in five parts
print('Accuracy is:',np.mean(cvs))


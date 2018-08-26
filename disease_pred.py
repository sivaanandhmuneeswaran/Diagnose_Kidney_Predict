# -*- coding: utf-8 -*-
"""
Created on Sat Aug 25 11:09:40 2018

@author: Sivaanandh Muneeswaran
@email: sivaanandhmuneeswaran@gmail.com

* The objective of this model is to predict if a person has chronic kidney disease
  based on around 25 metrics(age,blood pressure, etc)
The model is trained using Decision Tree
The obtained accuracy is around 99% (tested using accuracy_score module of sklearn.metrics)

"""
#Importing dataset
import pandas as pd
import numpy as np

dataset = pd.read_csv('chronic_kidney_disease_full.csv')
dataset = dataset.replace('?',np.nan)

X = dataset.iloc[:,0:24].values
y = dataset.iloc[:,24].values

#Data preprocessing
from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
X[:,5] = encoder.fit_transform(X[:,5].astype(str))

encoder = LabelEncoder()
X[:,6] = encoder.fit_transform(X[:,6].astype(str))

encoder = LabelEncoder()
X[:,7] = encoder.fit_transform(X[:,7].astype(str))

encoder = LabelEncoder()
X[:,8] = encoder.fit_transform(X[:,8].astype(str))

encoder = LabelEncoder()
X[:,9] = encoder.fit_transform(X[:,9].astype(str))

encoder = LabelEncoder()
X[:,18] = encoder.fit_transform(X[:,18].astype(str))

encoder = LabelEncoder()
X[:,19] = encoder.fit_transform(X[:,19].astype(str))

encoder = LabelEncoder()
X[:,20] = encoder.fit_transform(X[:,20].astype(str))

encoder = LabelEncoder()
X[:,21] = encoder.fit_transform(X[:,21].astype(str))

encoder = LabelEncoder()
X[:,22] = encoder.fit_transform(X[:,22].astype(str))

encoder = LabelEncoder()
X[:,23] = encoder.fit_transform(X[:,23].astype(str))

#Filling missing values using various strategy
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values="NaN",strategy ='mean',axis=0)
imputer = imputer.fit(X[:,np.r_[0:5,9:18]])
X[:,np.r_[0:5,9:18]] = imputer.transform(X[:,np.r_[0:5,9:18]])

imputer = Imputer(missing_values="NaN",strategy = 'most_frequent',axis=0)
imputer = imputer.fit(X[:,np.r_[5:9,18:24]])
X[:,np.r_[5:9,18:24]] = imputer.transform(X[:,np.r_[5:9,18:24]])

#Encoding categorical data
from keras.utils import np_utils
encoder = LabelEncoder()
encoder.fit(y)
encoded_y = encoder.transform(y)
y = np_utils.to_categorical(encoded_y)

#Splitting test and train data
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.3,random_state=0)

#Feature Scaling using MinMaxScalar
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)

#Fitting the data to DecisionTree
from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(criterion='entropy')
classifier.fit(X_train,y_train)

#Predicting the results
y_pred = classifier.predict(X_test)

#Calculating accuracy of model 
from sklearn.metrics import accuracy_score
accuracy_score(y_test,y_pred)

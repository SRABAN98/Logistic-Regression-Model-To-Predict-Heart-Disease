#import the necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


#import the dataset
dataset = pd.read_csv(r"C:\Users\dell\OneDrive\Documents\Data Science\28th,29th\TASK-24\framingham.csv")


#splitting the dataset into I.V and D.V
x = dataset.iloc[:,:-1].values
y = dataset.iloc[:,-1].values


#impute the missing values in the I.V of the dataset
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values = np.nan, strategy = "mean")
imputer.fit(x)
x = imputer.transform(x)


#splitting the dataset into training and test set
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.20, random_state = 0)


#feature scalling the dataset for the better performance
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
x_train = sc_x.fit_transform(x_train)
x_test = sc_x.fit_transform(x_test)


#training the Logistic Regression Model
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression()
classifier.fit(x_train, y_train)


#predicting the test set results
y_pred = classifier.predict(x_test)


#Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)


#This is to get the Models Accuracy 
from sklearn.metrics import accuracy_score 
ac = accuracy_score(y_test, y_pred)
print(ac)


#This is to get the Classification Report
from sklearn.metrics import classification_report
cr = classification_report(y_test, y_pred)
cr


#bias calculation
bias = classifier.score(x_train, y_train)
bias


#variance calculation
variance = classifier.score(x_test, y_test)
variance

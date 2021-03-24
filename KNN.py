# -- coding: utf-8 --
"""
Created on Tue Dec 15 04:00:29 2020

@author: Taha
"""


#Importing Libraries
from matplotlib import cm

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import *
import joblib 
import time
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt

t=time.time()
#Reading the dataset
app=pd.read_csv('Data1.csv').dropna()
app.head()

#Seperating input and output data
I=app[['I', 'TP','AP', 'RP']]
O=app['Appliance']

#Splitting the data into training and testing sets
(I_tr, I_ts, O_tr, O_ts)=train_test_split(I,O, test_size=0.1)


#Initialing the classifier model
model = KNeighborsClassifier(n_neighbors=5)

#Model Training
model.fit(I_tr,O_tr)
trainingTime=time.time()-t; #Time taken by model training
joblib.dump(model, 'KNNIden_Model_desk&bulb.pkl') #Saving Model

model=joblib.load('KNNIden_Model_desk&bulb.pkl') #loading saved model  

#Identifying appliances for Input tests set
pred=model.predict(I_ts)
IdenTime=time.time()-t;



#Determining the accuracy
print('Accuracy: %f' % accuracy_score(O_ts, pred))
print('Precision: %f' % precision_score(O_ts, pred, average='macro'))
print('Recall: %f' % recall_score(O_ts, pred, average='macro'))
print('F1 Score: %f' % f1_score(O_ts, pred, average='macro'))
model.score(I_tr, O_tr)
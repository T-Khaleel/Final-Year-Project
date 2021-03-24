# -*- coding: utf-8 -*-
"""
Created on Tue Jan 26 09:57:17 2021

@author: Taha
"""

#Importing Libraries

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import *
import time
import joblib 

t=time.time()
#Reading the dataset
app=pd.read_csv('Data1.csv').dropna()

#Seperating input and output data
I=app[['I', 'TP','AP', 'RP']]
O=app['Appliance']
#Splitting the data into training and testing sets
(I_tr, I_ts, O_tr, O_ts)=train_test_split(I,O, test_size=0.1)


#Initialing the classifier model
model = MLPClassifier()


#Model Training
model.fit(I_tr,O_tr)
trainingTime=time.time()-t; #Time taken by model training
joblib.dump(model, 'MLPIden_Model.pkl') #Saving Model


model=joblib.load('MLPIden_Model.pkl') #loading saved model  
#Identifying appliances for Input tests set
pred=model.predict(I_ts)
IdenTime=time.time()-t;

#Determining the accuracy
print('Accuracy: %f' % accuracy_score(O_ts, pred))
print('Precision: %f' % precision_score(O_ts, pred, average='macro'))
print('Recall: %f' % recall_score(O_ts, pred, average='macro'))
print('F1 Score: %f' % f1_score(O_ts, pred, average='macro'))
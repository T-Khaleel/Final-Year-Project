#Importing Libraries

import keras
import tensorflow as tf

from numpy import array
from keras.preprocessing.text import one_hot
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers.core import Activation, Dropout, Dense
from keras.layers import Flatten, LSTM
from keras.layers import GlobalMaxPooling1D
from keras.models import Model, load_model
from keras.layers.embeddings import Embedding
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras.layers import Input
from keras.layers.merge import Concatenate
from keras.layers import Bidirectional
import joblib
    
import pandas as pd
import numpy as np
import time
    
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import *
from sklearn.preprocessing import MinMaxScaler
import pickle

t=time.time()

#Reading the dataset and specifying inputs and outputs
dataset = pd.read_csv('Samples1.csv').dropna()
x1 = dataset['Current'].to_numpy()
x2 = dataset['True Power'].to_numpy()
x3 = dataset['Apparent Power'].to_numpy()
x4 = dataset['Reactive Power'].to_numpy()  
y=dataset.drop(columns=['Current','True Power', 'Apparent Power','Reactive Power', 'Appliance'])
x = np.column_stack((x1, x2, x3,x4))


#Splitting and Normalizing the dataset
(I_tr, I_ts, O_tr, O_ts)= train_test_split(x,y, test_size=0.25,random_state=42,shuffle=False)
a = MinMaxScaler().fit_transform(I_tr)
b = MinMaxScaler().fit_transform(I_ts)


#Building Neural Network
ann = tf.keras.models.Sequential()
ann.add(tf.keras.layers.Dense(units=230,activation='relu'))
ann.add(tf.keras.layers.Dense(units=190,activation='relu'))
ann.add(tf.keras.layers.Dense(units=1))
ann.compile(optimizer='adam', loss='mse', metrics = ['mean_absolute_error'])

ann.fit(a, O_tr['Iron TP'], epochs=200, validation_split=0.2, verbose=1, batch_size=64)
filepath='Iron1.h5'
ann.save(filepath)
pred = {}
pred1 = {}


filepath='Iron1.h5'
ann=load_model(filepath)
t=time.time()

#Making Appliance specific Prediction
pred['Iron TP']=ann.predict(b)
disaggregationTime=time.time()-t
pred1['Iron TP']=ann.predict(a)

#Printing average of R-Squared, MSE, RMSE, MAE of all models
#Checking bias and variance         
print(r2_score(O_ts['Iron TP'],pred['Iron TP']))
print(mean_squared_error(O_ts['Iron TP'].values,pred['Iron TP']))
print(np.sqrt(mean_squared_error(O_ts['Iron TP'].values,pred['Iron TP'])))
print(mean_absolute_error(O_ts['Iron TP'],pred['Iron TP']))


print(r2_score(O_tr['Iron TP'],pred1['Iron TP']))
print(mean_squared_error(O_tr['Iron TP'].values,pred1['Iron TP']))
print(np.sqrt(mean_squared_error(O_tr['Iron TP'].values,pred1['Iron TP'])))
print(mean_absolute_error(O_tr['Iron TP'],pred1['Iron TP']))
                          
#Plotting Actual vs diaggregated data for the first 500 samples
                          
xpoints = O_ts['Iron TP'].values
xpoints = xpoints[0:500,]
xpoints1 = pred['Iron TP']
xpoints1 = xpoints1[0:500,]

plt.plot(xpoints, 'o', label="Actual Iron Power Consumption")
plt.plot(xpoints1, '*', label="Disaggregated Iron Power Consumption")
plt.xlabel('Samples')
plt.ylabel('True Power')
plt.title('Power Consumption')

plt.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol= 1)
fig1 = plt.gcf()
plt.show()
plt.draw()
fig1.savefig('Iron.png', dpi=600,bbox_inches='tight')





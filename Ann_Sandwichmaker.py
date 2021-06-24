

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
    
from sklearn.model_selection import train_test_split
from sklearn.metrics import *
from sklearn.preprocessing import StandardScaler,MinMaxScaler,MaxAbsScaler,minmax_scale
import pickle
import matplotlib.pyplot as plt

t=time.time()
dataset = pd.read_csv('Appliance2.csv').dropna()
x1 = dataset['Current'].to_numpy()
x2 = dataset['True Power'].to_numpy()
x3 = dataset['Apparent Power'].to_numpy()
x4 = dataset['Reactive Power'].to_numpy()  
y=dataset.drop(columns=['Current','True Power', 'Apparent Power','Reactive Power', 'Appliance'])
x = np.column_stack((x1, x2, x3,x4))


plt.scatter(dataset['Current'],dataset['Sandwichmaker TP'])
plt.show( )
(I_tr, I_ts, O_tr, O_ts)= train_test_split(x,y, test_size=0.25,random_state=42)
a = MinMaxScaler().fit_transform(I_tr)
b = MinMaxScaler().fit_transform(I_ts)

ann = tf.keras.models.Sequential()


ann.add(tf.keras.layers.Dense(units=200,activation='relu'))
ann.add(tf.keras.layers.Dense(units=200,activation='relu'))
ann.add(tf.keras.layers.Dense(units=1))
ann.compile(optimizer='adam', loss='mse', metrics = ['mean_absolute_error'])
history=ann.fit(a, O_tr['Sandwichmaker TP'], epochs=800, validation_split=0.2, verbose=1, batch_size=64)

filepath='Sandwichmaker1.h5'
ann.save(filepath)
pred = {}
pred1 = {}


filepath='Sandwichmaker1.h5'
ann=load_model(filepath)
t=time.time()

pred['Sandwichmaker TP']=ann.predict(b)
disaggregationTime=time.time()-t

pred1['Sandwichmaker TP Train set']=ann.predict(a)

print('Test set')
print(r2_score(O_ts['Sandwichmaker TP'],pred['Sandwichmaker TP']))
print(mean_squared_error(O_ts['Sandwichmaker TP'].values,pred['Sandwichmaker TP']))
print(np.sqrt(mean_squared_error(O_ts['Sandwichmaker TP'].values,pred['Sandwichmaker TP'])))
print(mean_absolute_error(O_ts['Sandwichmaker TP'],pred['Sandwichmaker TP']))


print('Train set')
print(r2_score(O_tr['Sandwichmaker TP'],pred1['Sandwichmaker TP Train set']))
print(mean_squared_error(O_tr['Sandwichmaker TP'],pred1['Sandwichmaker TP Train set']))
print(np.sqrt(mean_squared_error(O_tr['Sandwichmaker TP'],pred1['Sandwichmaker TP Train set'])))
print(mean_absolute_error(O_tr['Sandwichmaker TP'],pred1['Sandwichmaker TP Train set']))

xpoints = O_ts['Sandwichmaker TP'].values
xpoints = xpoints[0:500,]
xpoints1 = pred['Sandwichmaker TP']
xpoints1 = xpoints1[0:500,]



plt.plot(xpoints, 'o', label="Actual Sandwichmaker Power Consumption")
plt.plot(xpoints1, '*', label="Disaggregated Sandwichmaker Power Consumption")
plt.xlabel('Samples')
plt.ylabel('True Power')
plt.title('Power Consumption')

plt.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol= 1)
fig1 = plt.gcf()
plt.show()
plt.draw()
fig1.savefig('Sandwichmaker.png', dpi=600,bbox_inches='tight')




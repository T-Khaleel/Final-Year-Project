
from numpy import array
from keras.preprocessing.text import one_hots
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers.core import Activation, Dropout, Dense
from keras.layers import Flatten, LSTM
from keras.layers import GlobalMaxPooling1D
from keras.models import Model
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
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestRegressor 
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import *
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import seaborn as sns

t=time.time()
dataset = pd.read_csv('Samples.csv').dropna()
x1 = dataset['Current'].to_numpy()
x2 = dataset['True Power'].to_numpy()
x3 = dataset['Apparent Power'].to_numpy()
x4 = dataset['Reactive Power'].to_numpy()  
y=dataset.drop(columns=['Current','True Power', 'Apparent Power','Reactive Power', 'Appliance'])
x = np.column_stack((x1, x2, x3,x4))


#Standardization of input data
scaler = StandardScaler()
xs = scaler.fit_transform(x)

#Splitting the data into training and testing sets
(I_tr, I_ts, O_tr, O_ts)= train_test_split(xs,y, test_size=0.25,random_state=42)
print ('Datasets generated')

#Defing the Model
model = RandomForestRegressor(n_estimators=10,random_state=42)

#Creating a list with Appliance(to be disaggregated from aggregated input) coulumn names 
appliances = list(dataset.columns.values[7:])
print(appliances)


#Function to train and save Appliance specific GRUs
t=time.time()

def randomForest_mult_apps():
#Loop to read one appliance column train that against aggregated input data and saving the model with appliance specific name    
    for app in appliances:
        start = time.time() #Reads PC time which is start time for training of model
        y_t = O_tr[app]    #Appliance wise data  
        model.fit(I_tr, y_t) #Model training
        joblib.dump(model, 'model_Forest_'+ str(app) +'.pkl', compress=1)  #Model Saving
    return True
randomForest_mult_apps()
trainTime=time.time()-t
#Function to import saved models and make predictions on test input
pred = {}
pred1={}
t=time.time()

def predictMulApp():
    for app in appliances:
        start = time.time()
        model=joblib.load('model_Forest_'+ str(app) +'.pkl') #Reading Appliance specific Model
        pred[app]=model.predict(I_ts)#Making Appliance specific Prediction
        pred1[app]=model.predict(I_tr)
    return pred   
        
mul_pred =predictMulApp()     #Calling Function for Multiple Predictions and saving Appliance specific predictions in list 

disaggregationTime=time.time()-t

r2=[]
mse=[]
rmse=[]
mae=[]

#Function to check R-Squared, MSE, RMSE, MAE for appliance specific predictions 

for app in appliances:
      r2.append(r2_score(O_ts[app], mul_pred[app]))
      mse.append(np.mean(np.square(mul_pred[app]-O_ts[app].values)))
      rmse.append(np.sqrt(np.mean(np.square(mul_pred[app]-O_ts[app].values))))
      mae.append(np.mean(np.abs(mul_pred[app]-O_ts[app].values)))
      
#Printing average of R-Squared, MSE, RMSE, MAE of all models
print('Average R-Squared of the model is:') 
print(np.mean(r2))
print('Average MAE of the model is:') 
print(np.mean(mae))
print('Average MSE of the model is:') 
print(np.mean(mse))
print('Average RMSE of the model is:') 
print(np.mean(rmse))

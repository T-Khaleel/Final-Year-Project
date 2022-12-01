
from numpy import array
from keras.preprocessing.text import one_hot
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
from sklearn.tree import DecisionTreeRegressor 
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import *
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import seaborn as sns

t=time.time()
dataset = pd.read_csv('Samples1.csv').dropna()
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
(I_tr, I_ts, O_tr, O_ts)= train_test_split(xs,y, test_size=0.25)
print ('Datasets generated')


#Creating a list with Appliance(to be disaggregated from aggregated input) coulumn names 
appliances = list(dataset.columns.values[7:])
print(appliances)

t=time.time()

#Function to train Appliance specific models and predicting applaince specific true and apparent powers
def decisiontree_reg_mult_apps():  
    for app in appliances:
        y_train =   O_tr[app].values
        clf = DecisionTreeRegressor()
        clf.fit(I_tr, y_train)
        joblib.dump(clf, 'model_dtr_'+ str(app) +'.pkl', compress=1)  #Model Saving
    return True

decisiontree_reg_mult_apps()
TrainTime=time.time()-t


pred = {}
t=time.time()

def predictMulApp():
    for app in appliances:
        clf=joblib.load('model_dtr_'+ str(app) +'.pkl') #Reading Appliance specific Model
        pred[app]=clf.predict(I_ts) #Making Appliance specific Prediction
    return pred  
        
mul_pred=predictMulApp()
disaggTime=time.time()-t

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


# -*- coding: utf-8 -*-
"""
Created on Tue Feb  2 10:10:08 2021

@author: Taha
"""

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
    from sklearn.preprocessing import StandardScaler
    import pickle
    
    
    t=time.time()
    
    Data=pd.read_csv('Data1.csv').dropna()
    
    x1 = Data['I'].to_numpy()
    x2 = Data['TP'].to_numpy()
    x3 = Data['AP'].to_numpy()
    x4 = Data['RP'].to_numpy()
    
    y=Data.drop(columns=['I','TP', 'AP','RP', 'Appliance'])
    x = np.column_stack((x1, x2, x3,x4))
    scaler = StandardScaler()
    xs = scaler.fit_transform(x)
    xs=array(xs).reshape(len(x),1,4)
    (I_tr, I_ts, O_tr, O_ts)= train_test_split(xs,y, test_size=0.1)
    print ('Datasets generated')
    
    
    # split into train and test sets
    
    
    
    model = Sequential()
    model.add(LSTM(100, activation='relu', return_sequences=True, input_shape=(1,4)))
    model.add(LSTM(100, activation='relu'))
    
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    print(model.summary())
    
    appliances = list(Data.columns.values[4:8])
    print(appliances)
    
    Training_Time={}
    def lstm_mult_apps():
        for app in appliances:
            start = time.time() 
            y_t = O_tr[app]    #Appliance wise data  
            model.fit(I_tr, y_t, epochs=255, validation_split=0.1, verbose=1, batch_size=60) #Model training
            filepath='lstm_ '+str(app)+'.h5'
            model.save(filepath)
            Training_Time[app]= time.time() - start  #Saves time taken by a model training
        return Training_Time
    
    lstm_mult_apps()
    
    
    #Function to import saved models and make predictions on test input
    pred = {}
    def predictMulApp():
        for app in appliances:
            start = time.time()
            filepath='lstm_ '+str(app)+'.h5'
            model=load_model(filepath)
            pred[app]=model.predict(I_ts) #Making Appliance specific Prediction
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
    
    
    #Plotting the comparison plots of predicted and original outputs
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.plot(O_ts['Bulb TP'].values, 'g', label="Actual Bulb Power Consumption")
    ax1.plot(mul_pred['Bulb TP'], 'r', label="Disaggregated Bulb Power Consumption")
    ax1.legend(prop={'size': 5})
    
    fig2 = plt.figure()
    ax2 = fig2.add_subplot(111)
    ax2.plot(O_ts['Desktop TP'].values, 'g', label="Actual Desktop Power Consumption")
    ax2.plot(mul_pred['Desktop TP'], 'r', label="Disaggregated Desktop Power Consumption")
    ax2.legend(prop={'size': 5})
    
    fig = plt.figure()
    ax3 = fig.add_subplot(111)
    ax3.plot(O_ts['Cooler TP'].values, 'g', label="Actual Cooler Consumption")
    ax3.plot(mul_pred['Cooler TP'], 'r', label="Disaggregated Cooler Consumption")
    ax3.legend(prop={'size': 5})
    
    fig = plt.figure()
    ax3 = fig.add_subplot(111)
    ax3.plot(O_ts['Sandwichmaker TP'].values, 'g', label="Actual Sandwich Maker Consumption")
    ax3.plot(mul_pred['Sandwichmaker TP'], 'r', label="Disaggregated Sandwich Maker Consumption")
    ax3.legend(prop={'size': 5})
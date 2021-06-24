
#Importing Libraries

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import *
import time
import joblib 
from sklearn.utils.multiclass import unique_labels
import seaborn as sns
import matplotlib.pyplot as plt     
t=time.time()
#Reading the dataset
app=pd.read_csv('Samples1.csv').dropna()
app.head()

#Seperating input and output data
I=app[['Current', 'True Power','Apparent Power', 'Reactive Power']]
O=app['Appliance']

#Splitting the data into training and testing sets
(I_tr, I_ts, O_tr, O_ts)=train_test_split(I,O, test_size=0.25,random_state=42)


#Initialing the classifier model
model = LogisticRegression(random_state=1000)
t=time.time()

#Model Training
model.fit(I_tr,O_tr)

trainingTime=time.time()-t; #Time taken by model training
joblib.dump(model, 'LogReg_Model.pkl') #Saving Model


model=joblib.load('LogReg_Model.pkl') #loading saved model  
#Identifying appliances for Input tests set
t=time.time()

pred=model.predict(I_ts)
IdenTime=time.time()-t;

#Determining the accuracy
print('Accuracy: %f' % accuracy_score(O_ts, pred))
print('Precision: %f' % precision_score(O_ts, pred, average='macro'))
print('Recall: %f' % recall_score(O_ts, pred, average='macro'))
print('F1 Score: %f' % f1_score(O_ts, pred, average='macro'))


cf_matrix = confusion_matrix(O_ts, pred)

ax= plt.subplot()
heatmap = sns.heatmap(cf_matrix, annot = True,fmt='g');  #annot=True to annotate cells, ftm='g' to disable scientific notation
heatmap.set_xticklabels(heatmap.get_xticklabels(), rotation=90) 
heatmap.set_yticklabels(heatmap.get_yticklabels(), rotation=0) 

# labels, title and ticks
ax.set_xlabel('Predicted labels');
ax.set_ylabel('True labels'); 
ax.set_title('Confusion Matrix'); 
ax.xaxis.set_ticklabels(unique_labels(O_ts))
ax.yaxis.set_ticklabels(unique_labels(O_ts));

fig = heatmap.get_figure()    
fig.savefig('Matrix_Logistic.png', dpi=600,bbox_inches='tight')


 
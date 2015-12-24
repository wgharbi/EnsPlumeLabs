# -*- coding: utf-8 -*-
"""
Created on Tue Dec 15 14:34:20 2015

@author: Hugo
"""

import pandas as pd
import numpy as np
from statistics import *
import csv
import networkx as nx

data_train = pd.read_csv("./train.csv")
writer = csv.writer(open("./labels.csv", 'w'))
data_labels = pd.read_csv("./validation.csv",delimiter=";")
data_test = pd.read_csv("./test.csv")

#%%
#Let's have a look at the data :
global_stats,stations_stats, feature_stats, features_available_station = stats(data_train)
plot_stats(feature_stats)
plot_stations(feature_stats)

#%% Let's remove the date 
data_train = data_train.iloc[:,1:]
data_test = data_test.iloc[:,1:]

#%%
from sklearn.linear_model import Ridge
from sklearn.cross_validation import train_test_split
from sklearn.metrics import mean_squared_error
X_train, X_test, y_train, y_test = train_test_split(data_train, data_labels, test_size=0.2, random_state=42)


clf=Ridge(solver='lsqr')
clf.fit(X_train,y_train)

#%% Let's have a look at our 96 predictions, in average
y_pred = clf.predict(X_test)
print "MISE : ", mean_squared_error(y_test,y_pred)

plt.figure()
plt.plot(np.arange(0,96),np.mean(y_test,axis=0),color="red",label="Average real value")
plt.plot(np.arange(0,96),np.mean(y_pred,axis=0),color="blue",label="Average predicted value")
plt.axvline(x=24,color="grey",linestyle="dashed")
plt.axvline(x=48,color="grey",linestyle="dashed")
plt.axvline(x=72,color="grey",linestyle="dashed")
plt.xlabel="Polutant"
plt.ylabel="Concentration"
plt.legend(loc="lower right")
plt.show()

#%%
y_pred_final = clf.predict(data_test)
y_pred_final = pd.DataFrame(y_pred_final, columns = data_labels.columns)
y_pred_final.to_csv()

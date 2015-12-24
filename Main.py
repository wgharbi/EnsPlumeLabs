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
import seaborn as sb

sb.set_context() #Let's make some beautiful graphs ! (optionnal)

data_train = pd.read_csv("./train.csv")
writer = csv.writer(open("./labels.csv", 'w'))
data_labels = pd.read_csv("./validation.csv",delimiter=";")
data_test = pd.read_csv("./test.csv")

#%%
#Let's have a look at the data :
global_stats,stations_stats, feature_stats, features_available_station = stats(data_train)
plot_stats(feature_stats)
plot_stations(feature_stats)

#%% Let's remove the date column
data_train = data_train.iloc[:,1:]
data_test = data_test.iloc[:,1:]

#%%
from sklearn.linear_model import Ridge
from sklearn.cross_validation import train_test_split

X_train, X_test, y_train, y_test = train_test_split(data_train, data_labels, test_size=0.2, random_state=42)


clf=Ridge(alpha=0.1,normalize=True,solver='lsqr')
clf.fit(X_train,y_train)


"""
Quelques pistes :
    - Faire une moyenne des mesures des 24 dernières heure par polluants et faire regression là-dessus, plutot qu'avec les valeurs de chaque station
    - Bien faire gaffe que la prediction finale se fait sur la station 04143, penser à donner des "poids" aux données de chaque station (plus pour 04143)
    - Autres regressions possibles
    - Cross valider et optimiser les hyper-paramètres.
    - Bien comprendre le MSE et le R2
    - Virer carrément des features ? (chi2 ?)
"""

#%% Let's have a look at our 96 predictions, in average
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error

y_pred = clf.predict(X_test)
print "MSE : ", mean_squared_error(y_test,y_pred)
print "R2 : ",r2_score(y_test,y_pred)

fig, ax = plt.subplots()
ax.plot(np.arange(0,96),np.mean(y_test,axis=0),color="red",label="Average real value")
ax.plot(np.arange(0,96),np.mean(y_pred,axis=0),color="blue",label="Average predicted value")
ax.axvline(x=24,color="grey",linestyle="dashed")
ax.axvline(x=48,color="grey",linestyle="dashed")
ax.axvline(x=72,color="grey",linestyle="dashed")
ax.set_xlabel('Polutant')
ax.set_ylabel("Concentration")
ax.legend(loc="lower right")
plt.show()

#%% Let's have a look at the weights, in average, of the regression coefficients
fig, ax = plt.subplots()
ax.bar(np.arange(0,3728),np.mean(coefs,axis=0), color = "blue")
ax.set_xlabel('Coefficent index')
ax.set_ylabel("Average value")
plt.show()

#%% Write the final solution for submission
y_pred_final = clf.predict(data_test)
y_pred_final = pd.DataFrame(y_pred_final, columns = data_labels.columns)
y_pred_final.to_csv("./result.csv",index=False,sep=";")

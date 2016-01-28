# -*- coding: utf-8 -*-
"""
Created on Tue Dec 15 14:34:20 2015

@author: Hugo
"""

import pandas as pd
import numpy as np
import csv
import seaborn as sb

from preprocessing import *
from visualization import *
from statistics import *
from postprocessing import *

sb.set_context() #Let's make some beautiful graphs ! (optional)

data_train = pd.read_csv("./train.csv")
writer = csv.writer(open("./labels.csv", 'w'))
data_labels = pd.read_csv("./validation.csv",delimiter=";")
data_test = pd.read_csv("./test.csv")

#%%
#Let's have a look at the data :
global_stats,stations_stats, feature_stats, features_available_station = stats(data_train)
plot_stats(feature_stats)
plot_stations(feature_stats)

#%% Plot the variations of polutants during the full period
plot_polutants_time_series(data_train)


#%% Edit 25/12 : new idea : check the correlation bteween the labels to be predicted (if not correlated create individual predictors...)

plot_correlation_matrix(data_labels,data_type="labels")

"""Interpretation of the results :
    Each square is a correlation matrix of 24 features to be predicted in the following
    order : PM2,PM10,O3,NO2
    - PM2 and PM10 are highly correlated to themselves on a 24h window (size of the squares)
    meaning that these polutants are less volatile and thus less prone to high variations
    than NO2 and 03. These last two are highly correlated to themselves for only few hours
    - PM2 and PM10 are highly correlated : their correlation squares are almost similar.
    At the same hour, they have approximataly the same value
    - O3 is not correlated to PMs and middly correlated to NO2
    - NO2 is middly correlated to the 3 
"""    
#%% Let's do the same thing with the input data

plot_correlation_matrix(data_train,data_type="features")
"""Note for Wiem : 
    Les données d'entrée sont super corrélées dans la mesure où le dataset est constitué 
    de jour glissants (ie, une ligne = la ligne précédente translatée d'1h), du coup, lignes
    et features sont fortement corrélés, en atteste la figure suivante, ce qui peut être 
    problématique pour la régression (générer de l'over-fitting)...
    """
data_to_train=scale(data_train,axis=0) #Rescale each columns
cov_mat=np.abs(np.cov(data_to_train.T))

plt.figure()
plt.hold('on')
plt.imshow(np.array(cov_mat))
plt.colorbar()
plt.tick_params(which='both', bottom='off', top='off', labelbottom='off')
plt.hold('off')
plt.show()

#Filter the lines repeating the same days to keep only different days
data_to_train2=scale(data_train.iloc[0:4031:24,:],axis=0) #Rescale each columns and take only different days
cov_mat2=np.abs(np.cov(data_to_train2.T))

plt.figure()
plt.hold('on')
plt.imshow(np.array(cov_mat2))
plt.colorbar()
plt.tick_params(which='both', bottom='off', top='off', labelbottom='off')
plt.hold('off')
plt.show()

"""Interpreting the results
Globally, the data columns are not really correlated, exept for a set of indicators : the wind indicators
This explains the repating diagonal black pattern (appears 17 times on a line because there are
17 stations measuring the wind speed and direction). This seems logical : if the stations are
not too far from each other, they will measure similar couples (wind speed, wind direction)

"""

#%% Let's remove the date column
data_train = data_train.iloc[:,1:]
data_test = data_test.iloc[:,1:]


from sklearn.cross_validation import train_test_split

X_train, X_test, y_train, y_test = train_test_split(data_train, data_labels, test_size=0.2, random_state=42)
y_test=y_test.values
y_train=y_train.values

#%% Other splitting without shuffling (to keep dates ordering)
data_train = data_train.iloc[:,1:]
data_test = data_test.iloc[:,1:]

nb_points=data_train.shape[0]
slice_index = round(0.8*nb_points)
X_train, X_test, y_train, y_test = data_train.iloc[:slice_index,:],data_train.iloc[slice_index:,:],data_labels.iloc[:slice_index,:],data_labels.iloc[slice_index:,:]

y_test=y_test.values
y_train=y_train.values

#%%
from sklearn.linear_model import Ridge
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
#%%
from sklearn.ensemble.forest import RandomForestRegressor
clf=RandomForestRegressor(n_estimators = 15,criterion='mse')
clf.fit(X_train,y_train)
#%%
from preprocessing import *


slicer = polutantSlicer(keep_wind=False)
slicer.fit(X_train)
PM2,PM10,O3,NO2 = slicer.transform(X_train)


#%% Let's have a look at our 96 predictions, in average
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from visualization import *

y_pred = clf.predict(X_test)
print "MSE : ", (np.sum(np.square(y_test-y_pred)))/((96))
print "R2 : ",r2_score(y_test,y_pred)
print "MSE sklearn : ", mean_squared_error(y_test,y_pred)

plot_average_regression(y_pred,y_test)

#%% Let's have a look at the weights, in average, of the regression coefficients

plot_regression_coefficient(clf,data_train)

#%% Let's have a look at the individual contributions of our regression to the total MSE (MSE per hour)
from postprocessing import *

plot_MSE_per_hour(y_pred,y_test)
"""
Ce genre de résultat pourrait être réutilisé dans un post-processing moyenneur "weighted"
Note :
    Les arbres sont plus robustes, en erreur, à l'heure de prédiction que les linear models.
"""

"""
ATTENTION WIEM :
    Mon post-processing à besoin d'avoir en entrée des lignes "triées" par date/heure comme le format initial et le format
    final. par conséquent il n'est pas possible (pour l'instant) d'effectuer de postprocessing sur y_pred et de tester les valeurs
    pour la raison que test_train_split mélange les examples d'apprentissage ! (donc perte de l'ordre dans l'ensemble
    de test, donc impossible de postprocess y_pred). 
"""


#%% Write the final solution for submission
y_pred_final = clf.predict(data_test)

plot_correlation_matrix(y_pred_final,data_type="labels")

y_pred_final_filtered = multi_polutant_averaged_postprocessing(y_pred.values,method="Backward average")

plot_correlation_matrix(y_pred_final_filtered,data_type="labels")

y_pred_final_filtered = pd.DataFrame(y_pred_final_filtered, columns = data_labels.columns)
y_pred_final_filtered.to_csv("./result.csv",index=False,sep=";")


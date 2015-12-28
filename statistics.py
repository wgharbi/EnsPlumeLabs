# -*- coding: utf-8 -*-
"""
Created on Tue Dec 15 15:54:48 2015

@author: Hugo
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx

#Get the basics stats about our problem

def stats(data):
    features = np.array(data.columns).astype(str)
    nb_features = len(features)
    feature_stats=pd.DataFrame()
    index=0
    for i in range(nb_features):
        name = features[i]
        indic_type = name.count('_')
        if indic_type==0:
            feature_type = name
            line = {'Station': 'global', 'Feature': feature_type, 'Hour':np.nan}

        
        if indic_type==1:
            feature_type,hour = name.split("_")
            line = {'Station': 'global', 'Feature': feature_type, 'Hour':np.nan}
            
        if indic_type==2:
            feature_type, station_number, hour = name.split("_")
            line = {'Station': station_number, 'Feature': feature_type, 'Hour':hour}
                 
        if indic_type==3:
            feature_type1, feature_type2, station_number, hour = name.split("_")
            line = {'Station': station_number, 'Feature': feature_type1, 'Hour':hour}
        """
        else :
            print name
        """
        line = pd.DataFrame(line,index=[index])
        feature_stats=feature_stats.append(line)
        index=index+1    
        #print 'indice est :',i 
    global_stats = feature_stats[feature_stats["Station"]=='global']
    stations_stats = feature_stats[feature_stats["Station"]!='global']
    print "---------------------------------"
    print "Statistics per station : "
    print stations_stats.describe()
    print ""
    print "---------------------------------"
    print "Global statistics : "
    print global_stats.describe()

    print "---------------------------------"   
    index=(feature_stats["Feature"].isin(['NO2']))
    filtered=feature_stats[index]
    print "Stations analysing NO2 : "
    print filtered["Station"].unique()
    print("")
    
    index=(feature_stats["Feature"].isin(['O3']))
    filtered=feature_stats[index]
    print "Stations analysing O3 : "
    print filtered["Station"].unique()
    print("")
    
    index=(feature_stats["Feature"].isin(['PM10']))
    filtered=feature_stats[index]
    print "Stations analysing PM10 : "
    print filtered["Station"].unique()
    print("")
    
    index=(feature_stats["Feature"].isin(['PM2']))
    filtered=feature_stats[index]
    print "Stations analysing PM2 : "
    print filtered["Station"].unique()
    print("")
    
    index=(feature_stats["Feature"].isin(['NO2',"PM2","O3","PM10"]))
    filtered=feature_stats[index]
    print "Features available for these 4 markers :"
    print filtered["Feature"].value_counts()
    
    
    features_available_station=[]
    station_names=feature_stats["Station"].unique().astype(str)
    for i in range(len(station_names)):
        temp = feature_stats[feature_stats["Station"]==station_names[i]]
        features_available=temp["Feature"].unique().astype(str)
        line=[station_names[i]]
        for j in range(len(features_available)): #loop over the features available for the station
            line.append(features_available[j])
        features_available_station.append(line)
        
        
    return global_stats,stations_stats, feature_stats, features_available_station
    
def plot_stats(feature_stats):
    plt.figure()
    feature_stats["Feature"].value_counts().plot(kind="bar")
    plt.xlabel="Type de feature"
    plt.ylabel="Nb disponibles"
    plt.show()
    plt.figure()
    feature_stats["Station"].value_counts().plot(kind="bar")
    plt.xlabel="N° de station"
    plt.ylabel="Nb de features disponibles"
    
    
    
    
def plot_stations(feature_stats):
    import  matplotlib.colors as cl
    import networkx as nx
    station_names=feature_stats["Station"].unique().astype(str)
    G=nx.Graph()
    pos=nx.spring_layout(G)
    color=[]
    for i in range(len(station_names)):
        G.add_node(station_names[i])
        G.add_edge(station_names[0],station_names[i])
        temp = feature_stats[feature_stats["Station"]==station_names[i]]
        features_available=len(temp["Feature"].unique())
        color.append(features_available)
    
    
    vmin = min(color)
    vmax = max(color)
    norm = cl.Normalize(vmin, vmax , clip = False)
    cmap=plt.cm.Blues
    
    
    fig=plt.figure(figsize=(10,6))
    nx.draw_networkx(G,node_color=color, node_size=2000, cmap=cmap,alpha=1,font_size=10)
    plt.xlabel="hello"
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm._A = []
    cb=plt.colorbar(sm,ticks=color)
    cb.set_label("Number of feature types available per station")
    plt.axis("off")
    plt.show()
        
def plot_polutants_time_series(data):
    #For PM10
    PM10_series = data[[
    "PM10_04099_0",
    "PM10_18053_0",
    "PM10_14012_0",
    "PM10_14033_0",
    "PM10_26016_0",
    "PM10_04004_0",
    "PM10_04031_0",
    "PM10_04034_0"]]
    
    PM2_series = data[[
    'PM2_5_18053_0',
    'PM2_5_14012_0',
    'PM2_5_14033_0',
    'PM2_5_04034_0']]
    
    O3_series = data[[
    'O3_04145_0',
    'O3_18008_0',
    'O3_18053_0',
    'O3_14012_0',
    'O3_14033_0',
    'O3_34017_0',
    'O3_34041_0',
    'O3_25040_0',
    'O3_26016_0',
    'O3_04004_0']]
    
    NO2_series = data[[
    'NO2_04105_0',
    'NO2_04141_0',
    'NO2_18008_0',
    'NO2_18053_0',
    'NO2_14012_0', 
    'NO2_14033_0', 
    'NO2_04059_0', 
    'NO2_26016_0', 
    'NO2_04004_0',
    'NO2_04031_0', 
    'NO2_04034_0']]
    
    #plot PM10
    average_PM10 = PM10_series.mean(axis=1)
    plt.figure()
    plt.hold("on")
    plt.plot(PM10_series, linewidth=1,alpha=0.4)
    plt.plot(average_PM10, linewidth=1, color = "red",label="Average PM10 across the stations")
    plt.legend()
    plt.xlabel = "Observed days"
    plt.hold("off")
    plt.show()
    
    #plot PM2
    average_PM2 = PM2_series.mean(axis=1)
    plt.figure()
    plt.hold("on")
    plt.plot(PM2_series, linewidth=1,alpha=0.4)
    plt.plot(average_PM2, linewidth=1, color = "red",label="Average PM2_5 across the stations")
    plt.legend()
    plt.hold("off")
    plt.show()
    
    #plot 03
    average_O3 = O3_series.mean(axis=1)
    plt.figure()
    plt.hold("on")
    plt.plot(O3_series, linewidth=1,alpha=0.4)
    plt.plot(average_O3, linewidth=1, color = "red",label="Average O3 across the stations")
    plt.legend()
    plt.hold("off")
    plt.show()
    
    #plot NO2
    average_NO2 = NO2_series.mean(axis=1)
    plt.figure()
    plt.hold("on")
    plt.plot(NO2_series, linewidth=1,alpha=0.4)
    plt.plot(average_NO2, linewidth=1, color = "red",label="Average NO2 across the stations")
    plt.legend()
    plt.hold("off")
    plt.show()
    
    return 
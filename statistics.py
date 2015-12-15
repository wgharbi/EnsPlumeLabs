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
    print "Statistiques sur les stations : "
    print stations_stats.describe()
    print ""
    print "---------------------------------"
    print "Statistiques globales : "
    print global_stats.describe()
    
        
    return global_stats,stations_stats, feature_stats
    
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
        
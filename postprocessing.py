# -*- coding: utf-8 -*-
"""
Created on Wed Jan 13 14:42:04 2016

@author: Hugo
"""
import numpy as np

def sum_diag(mat,i,j):
    n,p=mat.shape
    value = mat[i,j]
    index1=i
    index2=j
    s=1
    while index1 != n-1  and index2 !=0 :
        index1=index1+1
        index2=index2-1
        temp=value
        value=temp+mat[index1,index2]
        s=s+1
        #print "entree"
    
    
    value=value/s
    return value

#%%

def averaged_postprocessing(mat):
    n,p = mat.shape
    filtered = np.zeros((n,p))
    for i in range(n):
        for j in range(p):
            filtered[i,j] = sum_diag(mat,i,j)
            
    return filtered
        
def multi_polutant_averaged_postprocessing(y_pred):
    PM2_pred = y_pred[:,:24]
    PM10_pred = y_pred[:,24:48]
    O3_pred = y_pred[:,48:72]
    NO2_pred = y_pred[:,72:96]
    
    PM2_pred_filt = averaged_postprocessing(PM2_pred)
    PM10_pred_filt = averaged_postprocessing(PM10_pred)
    O3_pred_filt = averaged_postprocessing(O3_pred)
    NO2_pred_filt = averaged_postprocessing(NO2_pred)
    
    y_pred_filtered = np.hstack((PM2_pred_filt,PM10_pred_filt,O3_pred_filt,NO2_pred_filt))
    return y_pred_filtered
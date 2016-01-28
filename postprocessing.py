# -*- coding: utf-8 -*-

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
 

def averaged_sum_diag(mat,filt_mat,i,j,weights):
    n,p=mat.shape
    norm = sum(weights)
    value = weights[j]*mat[i,j]
    index1=i
    index2=j
    while index1 != n-1  and index2 !=0 :
        index1=index1+1
        index2=index2-1
        temp=value
        value=temp+weights[index2]*mat[index1,index2]
        
    value=value/norm
    print value
    filt_mat[i,j] = value
    index1=i
    index2=j
    while index1 != n-1  and index2 !=0 :
        index1=index1+1
        index2=index2-1
        filt_mat[index1,index2]=value
        print "entree2"
        
    return filt_mat
  #%%  
    
    
def get_diag(mat,i,j):
    n,p=mat.shape
    value = mat[i,j]
    index1=i
    index2=j
    while index1 != n-1  and index2 !=0 :
        index1=index1+1
        index2=index2-1
        value=mat[index1,index2]
        #print "entree"
    
    
    return value
    

#%%

def averaged_postprocessing(mat):
    n,p = mat.shape
    filtered = np.zeros((n,p))
    for i in range(n):
        for j in range(p):
            filtered[i,j] = sum_diag(mat,i,j)
            
    return filtered
    
def selective_postprocessing(mat):
    #Note : in reality this method is a weighted_averaged_postprocessing with weight equal to 1 on first column and 0 elsewhere
    n,p = mat.shape
    filtered = np.zeros((n,p))
    for i in range(n):
        for j in range(p):
            filtered[i,j] = get_diag(mat,i,j)
            
    return filtered
    
def weighted_averaged_postprocessing(mat,weights):
    n,p = mat.shape
    last_col_index = p-1
    filtered = np.zeros((n,p))
    #First let's deal with the left upper corner problems
    for i in range(n):
        if i == 0:
            for j in range(p):
                print "i = ",i
                print "j =",j
                filtered = averaged_sum_diag(mat,filtered,i,j,weights)
        else:
            filtered = averaged_sum_diag(mat,filtered,i,last_col_index,weights)
            
    return filtered
        
#%%        
def multi_polutant_averaged_postprocessing(y_pred, method, weights=None):
    PM2_pred = y_pred[:,:24]
    PM10_pred = y_pred[:,24:48]
    O3_pred = y_pred[:,48:72]
    NO2_pred = y_pred[:,72:96]
    
    if method=="Backward average":
        PM2_pred_filt = averaged_postprocessing(PM2_pred)
        PM10_pred_filt = averaged_postprocessing(PM10_pred)
        O3_pred_filt = averaged_postprocessing(O3_pred)
        NO2_pred_filt = averaged_postprocessing(NO2_pred)
        
    if method=="Selection":
        PM2_pred_filt = selective_postprocessing(PM2_pred)
        PM10_pred_filt = selective_postprocessing(PM10_pred)
        O3_pred_filt = selective_postprocessing(O3_pred)
        NO2_pred_filt = selective_postprocessing(NO2_pred)
        
    if method=="Weighted average":
        if weights == None :
            print "Please provide weights as input parameter"
            return
        else :
            PM2_pred_filt = weighted_averaged_postprocessing(PM2_pred,weights)
            PM10_pred_filt = weighted_averaged_postprocessing(PM10_pred,weights)
            O3_pred_filt = weighted_averaged_postprocessing(O3_pred,weights)
            NO2_pred_filt = weighted_averaged_postprocessing(NO2_pred,weights)
    
    y_pred_filtered = np.hstack((PM2_pred_filt,PM10_pred_filt,O3_pred_filt,NO2_pred_filt))
    return y_pred_filtered
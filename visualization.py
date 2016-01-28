# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import seaborn as sb
import numpy as np

def plot_correlation_matrix(data,data_type):
    from sklearn.preprocessing import scale
    import seaborn as sb
    
    if data_type== "labels":
        data_to_predict=scale(data,axis=0) #Rescale each columns
        cov_mat=np.abs(np.cov(data_to_predict.T))
        
        xlabels=12*[""]+["PM2_5"]+24*[""]+["PM10"]+24*[""]+["O3"]+24*[""]+["NO2"]
        ylabels=12*[""]+["PM2_5"]+24*[""]+["PM10"]+24*[""]+["O3"]+24*[""]+["NO2"]
        
        plt.figure()
        plt.hold('on')
        sb.heatmap(cov_mat,cmap="Greys",square=False,xticklabels=xlabels,yticklabels=ylabels)
        #plt.plot([96,0],[96,0],color="red",LineWidth=2)
        plt.title("Correlation between the features to predict (in the train dataset)")
        plt.hold('off')
        plt.show()
        
        """
        #Filter the lines repeating the same days to keep only different days
        data_to_predict2=scale(data.iloc[0:4031:24,:],axis=0) #Rescale each columns
        cov_mat2=np.abs(np.cov(data_to_predict2.T))
        
        plt.figure()
        plt.hold('on')
        plt.hold('on')
        sb.heatmap(cov_mat2,cmap="Greys",square=False,xticklabels=xlabels,yticklabels=ylabels)
        #plt.plot([96,0],[96,0],color="red",LineWidth=2)
        plt.title("Correlation between the features to predict after filtering (in the train dataset)")
        plt.hold('off')
        plt.show()
        """
        
    if data_type == "features":
        data_to_train=data.drop("date",axis=1)
        data_to_train=scale(data_to_train,axis=0) #Rescale each columns
        cov_mat=np.abs(np.cov(data_to_train.T))
        
        plt.figure()
        plt.hold('on')
        plt.imshow(np.array(cov_mat))
        plt.colorbar()
        plt.tick_params(which='both', bottom='off', top='off', labelbottom='off')
        plt.hold('off')
        plt.show()
        
        """
        #Filter the lines repeating the same days to keep only different days
        data_to_train2=data.drop("date",axis=1)
        data_to_train2=scale(data_to_train2.iloc[0:4031:24,:],axis=0) #Rescale each columns and take only different days
        
        cov_mat2=np.abs(np.cov(data_to_train2.T))
        
        plt.figure()
        plt.hold('on')
        plt.imshow(np.array(cov_mat2))
        plt.colorbar()
        plt.tick_params(which='both', bottom='off', top='off', labelbottom='off')
        plt.hold('off')
        plt.show()
        """
    

def plot_average_regression(y_pred,y_test):
    fig, ax = plt.subplots()
    plt.hold("on")
    ax.plot(np.arange(0,96),np.mean(y_test,axis=0),color="red",label="Average real value")
    ax.plot(np.arange(0,96),np.mean(y_pred,axis=0),color="blue",label="Average predicted value")
    ax.axvline(x=24,color="grey",linestyle="dashed")
    ax.axvline(x=48,color="grey",linestyle="dashed")
    ax.axvline(x=72,color="grey",linestyle="dashed")
    ax.set_xlabel('Polutant')
    ax.set_ylabel("Concentration")
    ax.legend(loc="lower right")
    plt.hold("off")
    plt.show()
    
def plot_regression_coefficient(reg,data_train):
    #This function will plot the coefficient importances if reg is an ensemble method
    #and the classical regression coefficents values if reg is a linear regressor    
    
    if hasattr(reg,"feature_importances_")==True:
        coefs=reg.feature_importances_
        n_coefs = coefs.shape[1]
        fig, ax = plt.subplots()
        ax.bar(np.arange(0,n_coefs),np.mean(coefs,axis=0), color = "blue")
        ax.set_xlabel('Feature index')
        ax.set_ylabel("Feature importance in the ensemble method")
        plt.show()
        #check the most important coeffs, in averaged
        threshold = 0.3
        important_coefs=np.abs(np.mean(coefs,axis=0))>threshold
        important_coefs_names=data_train.columns[important_coefs]
        print important_coefs_names
        
    if hasattr(reg,"coef_")==True:
        coefs=reg.coef_
        n_coefs = coefs.shape[1]
        fig, ax = plt.subplots()
        ax.bar(np.arange(0,n_coefs),np.mean(coefs,axis=0), color = "blue")
        ax.set_xlabel('Feature index')
        ax.set_ylabel("Coefficient value")
        plt.show()
        #check the most important coeffs, in averaged
        threshold = 0.3
        important_coefs=np.abs(np.mean(coefs,axis=0))>threshold
        important_coefs_names=data_train.columns[important_coefs]
        print important_coefs_names
        
        
def plot_coeff_importances(reg,data_train):
    X_columns = data_train.columns
    
    ordering = np.argsort(reg.feature_importances_)[::-1][:50]
    importances = reg.feature_importances_[ordering]
    feature_names = X_columns[ordering]
    x = np.arange(len(feature_names))
    
    fig,ax = plt.subplots(1,1,figsize=(6,6))
    ax.bar(x, importances)
    ax.set_xticks(x + 0.5)
    ax.set_xticklabels(feature_names, rotation=90, fontsize=8)
    ax.set_ylabel('Feature importance')        
    
def plot_MSE_per_hour(y_pred,y_test):
    from sklearn.metrics import mean_squared_error
    #This function plots, for each pollutant, the MSE made by the regression along the hours to predict (ie the successive columns)
        
    
    PM2_pred = y_pred[:,:24]
    PM10_pred = y_pred[:,24:48]
    O3_pred = y_pred[:,48:72]
    NO2_pred = y_pred[:,72:96]
    
    PM2_test = y_test[:,:24]
    PM10_test = y_test[:,24:48]
    O3_test = y_test[:,48:72]
    NO2_test = y_test[:,72:96]
    
    mse_PM2=mean_squared_error(PM2_pred,PM2_test,multioutput="raw_values")
    mse_PM10=mean_squared_error(PM10_pred,PM10_test,multioutput="raw_values")
    mse_O3=mean_squared_error(O3_pred,O3_test,multioutput="raw_values")
    mse_NO2=mean_squared_error(NO2_pred,NO2_test,multioutput="raw_values")
    x_vec = np.arange(0,24)
    
    plt.figure()
    plt.xlim(xmin=0,xmax=24)
    xmin,xmax=plt.xlim()
    plt.hold("on")
    plt.plot(x_vec,mse_PM2,"ob",label="MSE on PM2")
    plt.plot(x_vec,mse_PM10,"oc",label="MSE on PM10")
    plt.plot(x_vec,mse_O3,"or",label="MSE on O3")
    plt.plot(x_vec,mse_NO2,"og",label="MSE on NO2")
    plt.legend(loc=4,frameon=True,framealpha=1)
    ymin,ymax = plt.ylim()
    plt.hlines(y=np.mean(mse_PM2),xmin=xmin,xmax=xmax+1,linestyles="dashed",color="blue")
    plt.hlines(y=np.mean(mse_PM10),xmin=xmin,xmax=xmax+1,linestyles="dashed",color="cyan")
    plt.hlines(y=np.mean(mse_O3),xmin=xmin,xmax=xmax+1,linestyles="dashed",color="red")
    plt.hlines(y=np.mean(mse_NO2),xmin=xmin,xmax=xmax+1,linestyles="dashed",color="green")
    plt.xlabel("Hour")
    plt.ylabel("MSE")
    plt.title("MSE per hour of prediction")
    plt.hold("off")
    plt.show()
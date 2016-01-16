
import pandas as pd

class polutantSlicer :
    def __init__(self,keep_wind) :
        #These indexes are kept for slicing the input data on the type of polutants
        self.index_PM2 =[]
        self.index_PM10 =[]
        self.index_O3 =[]
        self.index_NO2 =[]
        """
        self.index_PM2_label =[]
        self.index_PM10_label =[]
        self.index_O3_label =[]
        self.index_NO2_label =[]
        """
        #User option : keep or remove the wind indicators
        self.wind = keep_wind
        
    def fit(self,X):
        #The fitting method updates the indexes
        self.index_PM2 = [c for c in X.columns if "PM2" in c]
        self.index_PM10 =[c for c in X.columns if "PM10" in c]
        self.index_O3 = [c for c in X.columns if "O3" in c]
        self.index_NO2 = [c for c in X.columns if "NO2" in c]
        """
        self.index_PM2_label = [c for c in Y.columns if "PM2" in c]
        self.index_PM10_label =[c for c in Y.columns if "PM10" in c]
        self.index_O3_label = [c for c in Y.columns if "03" in c]
        self.index_NO2_label = [c for c in Y.columns if "NO2" in c]
        """
    def transform(self,X):
        #The transform methods creates 4 tables of input data
        index_PM2 = self.index_PM2 
        index_PM10 = self.index_PM10 
        index_O3 = self.index_O3
        index_NO2 = self.index_NO2 
        """
        index_PM2_label = self.index_PM2_label 
        index_PM10_label = self.index_PM10_label 
        index_O3_label = self.index_O3_label
        index_NO2_label = self.index_NO2_label 
        """
        PM2 = X[index_PM2]
        PM10 = X[index_PM10]
        O3 = X[index_O3]
        NO2 = X[index_NO2]
        """
        PM2_label = Y[index_PM2_label]
        PM10_label = Y[index_PM10_label]
        O3_label = Y[index_O3_label]
        NO2_label = Y[index_NO2_label]
        """
        #Get the columns of the other interesting indicators
        cols = [c for c in X.columns if "PM2" not in c and "wind" not in c and "PM10" not in c and "O3" not in c and "NO2" not in c]
        other_indicators = X[cols]
        
        PM2 = pd.concat([PM2,other_indicators], axis=1)
        PM10 = pd.concat([PM10,other_indicators], axis=1)
        O3 = pd.concat([O3,other_indicators], axis=1)
        NO2 = pd.concat([NO2,other_indicators], axis=1)
        
        #!!!!POUR LE MOMENT LE VENT N'EST PAS PRIS EN COMPTE
        return PM2,PM10,O3,NO2
        
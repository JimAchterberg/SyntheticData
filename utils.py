# -*- coding: utf-8 -*-
"""
Created on Sat Nov  5 18:05:46 2022

@author: Jim Achterberg
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Oct 23 15:42:43 2022

@author: Jim Achterberg
"""


import numpy as np
import pandas as pd
from numpy import random as rd
from sklearn.datasets import make_spd_matrix
from scipy.stats import ortho_group as og
from scipy.stats import rv_discrete
from keras.utils.np_utils import to_categorical
from scipy.stats import t,ks_2samp
from statsmodels.tsa.stattools import adfuller

class MinMaxScaler():
    def __init__(self,lo,hi):
        self.lo = lo
        self.hi = hi
        
    def fit(self,d):
        self.d = d
        if len(d.shape)==3:
            self.min_ = np.min(d,axis=(0,1))
            self.max_ = np.max(d,axis=(0,1))
        elif len(d.shape)==2:
            self.min_ = np.min(d,axis=0)
            self.max_ = np.max(d,axis=0)
        
            
    def transform(self):
        self.d = np.divide(np.subtract(self.d,self.min_),self.max_-self.min_)*\
            (self.hi-self.lo)+self.lo
        return self.d
    
    def rev_transform(self):
        self.d = np.add(np.multiply((self.d-self.lo)/(self.hi-self.lo),\
                                    (self.max_-self.min_)),self.min_)
        return self.d
    
class Standardizer():
    def __init__(self):
        pass
        
    def fit(self,d):
        self.d = d
        if len(d.shape)==3:
            self.mu = np.mean(self.d,axis=(0,1))
            self.std = np.std(self.d,axis=(0,1))
        elif len(d.shape)==2:
            self.mu = np.mean(self.d,axis=0)
            self.std = np.std(self.d,axis=0)
        
    def transform(self):
        self.d = np.divide(np.subtract(self.d,self.mu),self.std)
        return self.d
    
    def rev_transform(self):
        self.d = np.add(np.multiply(self.d,self.std),self.mu)
        return self.d
    
    

#function one hot encodes and appends at the end
def one_hot_encode(d,cat_columns):
    if len(d.shape)==3:
        for j in cat_columns:
            one_hot = to_categorical(d[:,:,j])
            d = np.delete(d,j,axis=2)
            d = np.concatenate((d,one_hot),axis=2)
    elif len(d.shape)==2:
        for j in cat_columns:
            one_hot = to_categorical(d[:,j])
            d = np.delete(d,j,axis=1)
            d = np.concatenate((d,one_hot),axis=1)
    return d
        
    
def concat_feat_attr(feat,attr):
    N,T,F = feat.shape
    _,A = attr.shape
    attributes = np.ones((N,T,A))*np.expand_dims(attr,axis=1)
    data = np.concatenate((feat,attributes),axis=2)
    return data


class tests():
    def __init__(self):
        pass
    
    def km_sm(self,x1,x2):
        stat,pv = ks_2samp(x1,x2,alternative='two-sided')
        return stat,pv
    
    def ADF(self,data):
        adf = adfuller(data)
        stat = adf[0]
        pv = adf[1]
        return stat,pv
    
    def DM_univariate(self,actuals,pred1,pred2,h=1,crit="MSE"):
        actuals = np.array(actuals)
        pred1 = np.array(pred1)
        pred2 = np.array(pred2)
        
        #amount of forecasts
        T = int(actuals.shape[0])
        
        # construct d according to crit
        if (crit == "MSE"):
            e1 = (actuals-pred1)**2
            e2 = (actuals-pred2)**2
            d = e1-e2
        elif (crit == "MAD"):
            e1 = abs(actuals-pred1)
            e2 = abs(actuals-pred2)
            d = e1-e2
        elif (crit == "MAPE"):
            e1 = (actuals-pred1)/actuals
            e2 = (actuals-pred2)/actuals
            d = e1-e2
            
        # Mean of d  
        mean_d = np.mean(d)
        
        # Find autocovariance and construct DM test statistics
        def autocovariance(Xi, N, k, Xs):
            autoCov = 0
            T = float(N)
            for i in np.arange(0, N-k):
                  autoCov += ((Xi[i+k])-Xs)*(Xi[i]-Xs)
            return (1/(T))*autoCov
        gamma = []
        for lag in range(0,h):
            gamma.append(autocovariance(d,d.shape[0],lag,mean_d)) # 0, 1, 2
        V_d = (gamma[0] + 2*sum(gamma[1:]))/T
        DM_stat=V_d**(-0.5)*mean_d
        harvey_adj=((T+1-2*h+h*(h-1)/T)/T)**(0.5)
        DM_stat = harvey_adj*DM_stat
        # Find p-value
        p_value = 2*t.cdf(-abs(DM_stat), df = T - 1)
        
        return DM_stat,p_value,d



    
    def DM_multivariate(self,actuals,pred1,pred2,h=1,crit="MSE"):
        #vectorized DM test. takes l1 norm in loss differential.
        actuals = np.array(actuals)
        pred1 = np.array(pred1)
        pred2 = np.array(pred2)
        
        #amount of forecasts
        T = int(actuals.shape[0])
        
        # construct d according to crit
        if (crit == "MSE"):
            e1 = (actuals-pred1)**2
            e2 = (actuals-pred2)**2
            d = np.linalg.norm(e1,1,axis=1)-np.linalg.norm(e2,1,axis=1)
        elif (crit == "MAD"):
            e1 = abs(actuals-pred1)
            e2 = abs(actuals-pred2)
            d = np.linalg.norm(e1,1,axis=1)-np.linalg.norm(e2,1,axis=1)
        elif (crit == "MAPE"):
            e1 = (actuals-pred1)/actuals
            e2 = (actuals-pred2)/actuals
            d = np.linalg.norm(e1,1,axis=1)-np.linalg.norm(e2,1,axis=1)
            
        # Mean of d  
        mean_d = np.mean(d)
        
        # Find autocovariance and construct DM test statistics
        def autocovariance(Xi, N, k, Xs):
            autoCov = 0
            T = float(N)
            for i in np.arange(0, N-k):
                  autoCov += ((Xi[i+k])-Xs)*(Xi[i]-Xs)
            return (1/(T))*autoCov
        gamma = []
        for lag in range(0,h):
            gamma.append(autocovariance(d,d.shape[0],lag,mean_d)) # 0, 1, 2
        V_d = (gamma[0] + 2*sum(gamma[1:]))/T
        DM_stat=V_d**(-0.5)*mean_d
        harvey_adj=((T+1-2*h+h*(h-1)/T)/T)**(0.5)
        DM_stat = harvey_adj*DM_stat
        # Find p-value
        p_value = 2*t.cdf(-abs(DM_stat), df = T - 1)
        
        return DM_stat,p_value,d

    
    
    

    

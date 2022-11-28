# -*- coding: utf-8 -*-
"""
Created on Sun Nov  6 23:20:31 2022

@author: Jim Achterberg
"""
import numpy as np
import pandas as pd
from numpy import random as rd
from sklearn.datasets import make_spd_matrix
from scipy.stats import ortho_group as og
from scipy.stats import rv_discrete


def VAR_gaussian(shape):
    N,T,F = shape
    sigma = make_spd_matrix(n_dim=F)
    err_mean = np.zeros(shape=F)
    errors = rd.multivariate_normal(err_mean,sigma,size=(N,T))
    V = rd.uniform(low=0,high=1,size=(F)) * np.identity(n=F)
    Q = og.rvs(dim=F)
    A = np.matmul(np.matmul(Q,V),np.linalg.inv(Q))
    data = np.empty((N,T,F))
    for t in range(T):
        if t==0:
            data[:,t,:] = errors[:,t,:]
        else:
            data[:,t,:] = np.matmul(errors[:,t-1,:],A)+errors[:,t,:]
    return data

def attributes(shape,n_cat,probs):
    N,A = shape
    vals = list(range(n_cat))
    distr = rv_discrete(a=0,b=n_cat-1,values=(vals,probs)) 
    attr = distr.rvs(size=(N,A))
    return attr

data = attributes((100,5),5,[.2]*5)



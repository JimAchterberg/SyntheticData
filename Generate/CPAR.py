# -*- coding: utf-8 -*-
"""
Created on Tue Oct 11 22:21:50 2022

@author: Jim Achterberg  
"""

import os 
from sdv.timeseries import PAR
from data_loading import data_cpar
import numpy as np
import pandas as pd

dataset = 'nmm'
df,entity_column,context_columns = data_cpar(dataset)
seq_len = 50
N = int(len(df)/seq_len)
A = len(context_columns)
F = df.shape[1]-len(context_columns)-len(entity_column)

model=PAR(entity_columns=entity_column,context_columns=context_columns,\
             epochs=10,sample_size=1,cuda=False,verbose=True)
model.fit(df)
syn = model.sample(N)

#return nd numpy arrays of feat and attr
def split_df(df,entity_column):
    df.drop(entity_column,axis=1,inplace=True)
    df = df.to_numpy().reshape(N,seq_len,F+A)
    df = df.astype(float)
    attr = df[:,0,list(range(F,F+A))]
    feat = np.delete(df,list(range(F,F+A)),axis=2)
    return feat,attr

syn_feat,syn_attr = split_df(syn,entity_column)
feat,attr = split_df(df,entity_column)

save_path = os.path.join(os.getcwd(),'Data','Synthetic')
file = os.path.join(save_path, f'cpar_{dataset}.npz')
np.savez(file,real_feat=feat,real_attr=attr,syn_feat=syn_feat,syn_attr=syn_attr)


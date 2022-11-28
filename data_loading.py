import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import LabelEncoder
from gretel_synthetics.timeseries_dgan.config import OutputType

def data_dgan(dataset):
    #dataset must be one of provided datasets, or preprocess your own dataset
    if dataset != 'nmm' and dataset!= 'cd':
        raise Exception('dataset must be one of the provided datasets, or preprocess your own.')
        
    data_path = os.path.join(os.getcwd(),'Data','Real',dataset+'.csv')
    data = pd.read_csv(data_path)
    
    if dataset=='nmm':
        #label encode categoricals
        data.typeStudy = LabelEncoder().fit_transform(data.typeStudy)
        
        #find sequence lengths
        seq_lens = data.groupby('PK',axis=0).count().iloc[:,0].to_numpy()
        n_seq = len(seq_lens)
        #drop sequences under length 50 and cutoff sequences at 50
        for i in range(n_seq):
            if seq_lens[i]<50:
                data.drop(data[data.PK==i].index,inplace=True)
            if seq_lens[i]>50:
                data.drop(data[data.PK==i].index[50:],inplace=True)
        n_seq = len(set(data.PK))
        
        #select desired features and attributes
        attr = pd.concat((data.Age,data.BMI,data.Sex,data.typeStudy),axis=1)
        feat = pd.concat((data.ExpSevo,data.InspSevo,data.TOF,data.Count,\
                          data.Esmeron,data.Temp,data.T1,data.Bridion),axis=1)
            
        #create nd numpy arrays of features and attributes
        attr = attr.to_numpy().reshape(n_seq,50,attr.shape[1])[:,0,:]
        feat = feat.to_numpy().reshape(n_seq,50,feat.shape[1])
        
        #all features are continuous, attributes are mixed
        feature_types = []
        attribute_types = []
        for i in range(feat.shape[2]):
            feature_types.append(OutputType.CONTINUOUS)
        attribute_types.append(OutputType.CONTINUOUS)
        attribute_types.append(OutputType.CONTINUOUS)
        attribute_types.append(OutputType.DISCRETE)
        attribute_types.append(OutputType.DISCRETE)
        
    elif dataset=='cd':
        #drop unnecessary columns 
        data.drop('Unnamed: 0',axis=1,inplace=True)
        
        #encode categoricals
        data.sex = LabelEncoder().fit_transform(data.sex)
        data.treat = LabelEncoder().fit_transform(data.treat)
        
        #find seq lens
        seq_lens = data.groupby(['site','id']).count().iloc[:,0].to_numpy()
        n_seq = len(seq_lens)
        #add IDs 
        curr=0
        id = 0
        ID=np.empty(shape=data.shape[0])
        for i in seq_lens:
            ID[curr:curr+i] = id
            id+=1
            curr+=i
        data.insert(0,'ID',ID)
        
      
        #drop undesired seq lens
        for i in range(n_seq):
            if seq_lens[i]<5:
                data.drop(data[data.ID==i].index,inplace=True)
            if seq_lens[i]>5:
                data.drop(data[data.ID==i].index[5:],inplace=True)
        n_seq = len(set(data.ID))
        
        #select desired features
        feat = np.expand_dims(data.twstrs.to_numpy(),axis=1)
        attr = pd.concat((data.age,data.treat,data.sex),axis=1).to_numpy()
        
        #create nd numpy arrays of features and attributes
        attr = attr.reshape(n_seq,5,attr.shape[1])[:,0,:]
        feat = feat.reshape(n_seq,5,feat.shape[1])
            
        
        feature_types = []
        attribute_types = []
        
        feature_types.append(OutputType.CONTINUOUS)
        attribute_types.append(OutputType.CONTINUOUS)
        attribute_types.append(OutputType.DISCRETE)
        attribute_types.append(OutputType.DISCRETE)
    return feat,attr,feature_types,attribute_types
        

def data_cpar(dataset):
    if dataset != 'nmm' and dataset!= 'cd':
        raise Exception('dataset must be one of the provided datasets, or preprocess your own.')
    data_path = os.path.join(os.getcwd(),'Data','Real',dataset+'.csv')
    data = pd.read_csv(data_path)
        
    if dataset=='nmm':
        #label encode categoricals
        #label encode categoricals
        data.typeStudy = LabelEncoder().fit_transform(data.typeStudy)
        
        #find sequence lengths
        seq_lens = data.groupby('PK',axis=0).count().iloc[:,0].to_numpy()
        n_seq = len(seq_lens)
        #drop sequences under length 50 and cutoff sequences at 50
        for i in range(n_seq):
            if seq_lens[i]<50:
                data.drop(data[data.PK==i].index,inplace=True)
            if seq_lens[i]>50:
                data.drop(data[data.PK==i].index[50:],inplace=True)
        n_seq = len(set(data.PK))
        
        #set correct entity and context columns
        data.PK = data.PK.astype(str)
        entity_column = ['PK']
        context_columns=['Age','BMI','Sex','typeStudy']
        df = pd.concat((data.PK,data.ExpSevo,data.InspSevo,data.TOF,data.Count,\
                          data.Esmeron,data.Temp,data.T1,data.Bridion,data.Age,\
                              data.BMI,data.Sex,data.typeStudy),axis=1)

    elif dataset=='cd':
        #drop unnecessary columns 
        data.drop('Unnamed: 0',axis=1,inplace=True)
        
        #encode categoricals
        data.sex = LabelEncoder().fit_transform(data.sex)
        data.treat = LabelEncoder().fit_transform(data.treat)
        
        #find seq lens
        seq_lens = data.groupby(['site','id']).count().iloc[:,0].to_numpy()
        n_seq = len(seq_lens)
        #add IDs 
        curr=0
        id = 0
        ID=np.empty(shape=data.shape[0])
        for i in seq_lens:
            ID[curr:curr+i] = id
            id+=1
            curr+=i
        data.insert(0,'ID',ID)
      
        #drop undesired seq lens
        for i in range(n_seq):
            if seq_lens[i]<5:
                data.drop(data[data.ID==i].index,inplace=True)
            if seq_lens[i]>5:
                data.drop(data[data.ID==i].index[5:],inplace=True)
        n_seq = len(set(data.ID))
        
        #set correct entity and context columns
        data.ID = data.ID.astype(str)
        entity_column = ['ID']
        context_columns = ['age','treat','sex']
        df = pd.concat((data.ID,data.twstrs,data.age,data.treat,data.sex),axis=1)
    
    return df,entity_column,context_columns


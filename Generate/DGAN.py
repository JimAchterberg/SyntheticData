import os 
from gretel_synthetics.timeseries_dgan.dgan import DGAN
from gretel_synthetics.timeseries_dgan.config import DGANConfig,Normalization
import numpy as np
from data_loading import data_dgan

dataset = 'nmm'
feat,attr,feature_types,attribute_types = data_dgan(dataset)
N,T,F = feat.shape

#run dgan 
config = DGANConfig(
apply_feature_scaling = True,
attribute_num_layers = 1,
max_sequence_len = T,
normalization=Normalization.ZERO_ONE,
sample_len = 5,
attribute_num_units = 100,
feature_num_layers = 1,
feature_num_units = 100,
use_attribute_discriminator = True,
apply_example_scaling = False,
batch_size = 30,
epochs = 10,
cuda=False 
)

model = DGAN(config)
model.train_numpy(features=feat,feature_types=feature_types,attributes=attr,attribute_types=attribute_types)
syn_attr,syn_feat = model.generate_numpy(N)

save_path = os.path.join(os.getcwd(),'Data','Synthetic')
file = os.path.join(save_path, f'dgan_{dataset}.npz')
np.savez(file,real_feat=feat,real_attr=attr,syn_feat=syn_feat,syn_attr=syn_attr)
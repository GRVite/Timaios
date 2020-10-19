#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  7 17:31:46 2020

@author: vite
"""


import os
import numpy as np
import pandas as pd
import neuroseries as nts
#from pylab import *
import matplotlib.pyplot as plt
from functions import *
from scipy.ndimage import gaussian_filter
from scipy.signal import correlate2d

import pandas as pd
import scipy.io
import numpy as np
# Should not import fonctions if already using tensorflow for something else
import sys, os
import itertools
import cPickle as pickle
import pandas as pd
from sklearn.model_selection import KFold
import xgboost as xgb
from sklearn.linear_model import LinearRegression


"""
A. LOAD DATA
"""

#rootDir = '/media/3TBHDD/Data'
rootDir = '/Users/vite/navigation_system/Data'
ID = 'A3304'
session = 'A3304-191127'
wakepos=0
events = ['0'] 

data_directory = rootDir + '/' + ID + '/' + session + '/' + session
if os.path.exists(data_directory+'/plots')==False:
    os.mkdir(data_directory+'/plots')
path =rootDir + '/' + ID + '/' + session
#count number of sessions
ns = int([i for i in os.listdir(path) if os.path.isdir(path+'/'+i)==True][-1][-1:])
if ns == 1:
    episodes=['wake']
else:
    episodes = ['wake' if i==wakepos else 'sleep' for i in list(range(ns+1))]


files = os.listdir(data_directory) 
from wrappers import loadSpikeData
spikes, shank = loadSpikeData(data_directory)
from wrappers import loadXML
n_channels, fs, shank_to_channel = loadXML(data_directory)

# Now we can load the position and rotation contained into the file Tracking_data.csv
# The order is by default [rotation y, rotation x, rotation z, position x, position y, position z] 
from wrappers import loadPosition
position = loadPosition(data_directory, events, episodes, n_ttl_channels = 1, optitrack_ch = 0)


from wrappers import loadEpoch
wake_ep                             = loadEpoch(data_directory, 'wake', episodes)
if ns > 1:
    sleep_ep                             = loadEpoch(data_directory, 'sleep')                    

#




###
#Place fields
###
GF, ext = computePlaceFields(spikes, position[['x', 'z']], wake_ep, 50)
plt.figure(figsize=(50,60))
for i,k in enumerate(GF.keys()):
    plt.subplot(5,raws+1,i+1)    
    tmp = gaussian_filter(GF[k].values, sigma = 1.3)
    im=imshow(tmp, extent = ext, cmap = 'jet', interpolation = 'bilinear')
    #plt.colorbar(im,fraction=0.046, pad=0.04)
    plt.title(str([i]))
    plt.xticks([]),plt.yticks([])
plt.show()
plt.suptitle('Place fields ' + session)

"""
DECODING
"""
neurons = [2, 3, 4, 6, 7, 16, 17, 25]

grid_fields = np.array([GF[n].values for n in neurons])

#bin firing rates
bin_size = 200
bins = np.arange(wake_ep.as_units('ms').start.iloc[0], wake_ep.as_units('ms').end.iloc[-1]+bin_size, bin_size)
spike_counts = pd.DataFrame(index = bins[0:-1]+np.diff(bins)/2, columns = neurons)
for i in neurons:
	spks = spikes[i].as_units('ms').index.values
	spike_counts[i], _ = np.histogram(spks, bins)
rate = np.sqrt(spike_counts/(bin_size*1e-3))
rate = rate.values

mean_firing = computeMeanFiringRate(spikes, [wake_ep], ['wake'])
mean_firing = mean_firing.loc[neurons].values.astype(np.float32)

# Normalizing by mean firing rate
rate = rate/mean_firing.flatten()




        
# ########################################################################
# # LEARNING ONE XGB
# ########################################################################    
X = rate


index         = np.digitize(position.as_units('ms').index.values, bins)
tmp         = position.groupby(index).mean()
tmp.index   = pd.Index(spike_counts.index.values)


Y = tmp[['x', 'z']].values

# order is [x, y]    
nbins_xy = 10
index = np.arange(nbins_xy*nbins_xy).reshape(nbins_xy,nbins_xy)
# binning pos

xpos = Y[:,0]
ypos = Y[:,1]

xbins = np.linspace(np.min(xpos[~np.isnan(xpos)]), np.max(xpos[~np.isnan(xpos)])+1e-6, nbins_xy)
ybins = np.linspace(np.min(ypos[~np.isnan(ypos)]), np.max(ypos[~np.isnan(ypos)])+1e-6, nbins_xy)


xposindex = np.digitize(Y[:,0], xbins).flatten()-1
yposindex = np.digitize(Y[:,1], ybins).flatten()-1
# setting class from index
clas = np.zeros(Y.shape[0])
for i in range(Y.shape[0]): clas[i] = index[xposindex[i],yposindex[i]]

dtrain = xgb.DMatrix(X, label=clas)        

params = {'objective': "multi:softprob",
'eval_metric': "mlogloss", #loglikelihood loss
'seed': 2925, #for reproducibility
'silent': 1,
'learning_rate': 0.05,
'min_child_weight': 0, 
'n_estimators': 1000,
# 'subsample': 0.5,
'max_depth': 4, 
'gamma': 0.5,
'num_class':index.max()+1}
num_round = 100

bst = xgb.train(params, dtrain, num_round)

dtest = xgb.DMatrix(X)
    
ymat = bst.predict(dtest)

pclas = np.argmax(ymat, 1)
x, y = np.mgrid[0:nbins_xy,0:nbins_xy]
clas_to_index = np.vstack((x.flatten(), y.flatten())).transpose()
Yp = clas_to_index[pclas]
# returning real position
pred_pos = np.zeros(Yp.shape)    
xx = xbins[0:-1] + np.diff(xbins)/2
yy = xbins[0:-1] + np.diff(ybins)/2
pred_pos[:,0] = xx[Yp[:,0]]
pred_pos[:,1] = yy[Yp[:,1]]


    



# return x[np.argmax(ymat,1)]        


#     sys.exit()
#     from pylab import *
#     figure()
#     plot(data['ang'].values, label = 'real')
#     plot(y_hat[:,0], label = 'pred')
#     legend()

#     figure()
#     plot(data['x'].values, data['y'].values, label = 'real')
#     plot(y_hat[:,1], y_hat[:,2], label = 'pred')
#     legend()

#     show()

#     sys.exit()
#     y_hat = test_decodage(features, targets, methods)            
#     results[k] = y_hat
#     score[k] = {}
#     y = data['ang'].values
#     for m in methods:
#         tmp = np.abs(y_hat[m]-y)
#         tmp[tmp>np.pi] = 2*np.pi - tmp[tmp>np.pi]
#         score[k][m] = np.sum(tmp)

    
# final_data[ses] = {}
# final_data[ses]['wake'] = {'score':score, 'output':results}

        sys.exit()
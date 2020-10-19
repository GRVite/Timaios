#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  9 20:20:44 2020

@author: vite
"""

"""
Created on Mon Sep  7 14:50:21 2020

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


###
#Place fields
###
GF, ext = computePlaceFields(spikes, position[['x', 'z']], wake_ep, 50)


"""
DECODING
"""
neurons = [2, 3, 4, 6, 7, 16, 17, 25]

grid_fields = np.array([GF[n].values for n in neurons])

#bin firing rates
bin_size = 200 #ms
bins = np.arange(wake_ep.as_units('ms').start.iloc[0], wake_ep.as_units('ms').end.iloc[-1]+bin_size, bin_size)
spike_counts = pd.DataFrame(index = bins[0:-1]+np.diff(bins)/2, columns = neurons)
for i in neurons:
	spks = spikes[i].as_units('ms').index.values
	spike_counts[i], _ = np.histogram(spks, bins)
rate = np.sqrt(spike_counts/(bin_size*1e-3))
rate = rate.values

# Normalizing by mean firing rate
mean_firing = computeMeanFiringRate(spikes, [wake_ep], ['wake'])
mean_firing = mean_firing.loc[neurons].values.astype(np.float32)
rate = rate/mean_firing.flatten()
grid_fields = np.array([grid_fields[i]/mean_firing[i] for i in range(len(neurons))])

linear_sum = np.dot(rate, grid_fields.reshape(len(neurons), np.product(grid_fields.shape[1:])))
filt_linear_sum = gaussian_filter(linear_sum, sigma = 3)


#Calculate alphas
xpred = []
ypred = []
for i in range(len(filt_linear_sum)-1):
    cc = correlate2d(filt_linear_sum[i+1], filt_linear_sum[i], mode = 'same')
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(cc)
    xpred.append(max_loc[0])
    ypred.append(max_loc[1])
xpred = np.array(xpred)
ypred = np.array(ypred)
# alphas = np.arctan2(ypred, xpred)
alphas = np.arctan2(ypred-filt_linear_sum.shape[2]//2, xpred-filt_linear_sum.shape[1]//2)

alphas [alphas<0] += 2*np.pi


#Take theta
theta = position.loc[:, 'ry'].as_units('ms').values
ind = position.as_units('ms').index
index=np.digitize(ind,bins)
dft = pd.DataFrame(index = index)
dft['theta']=theta
dft=dft.groupby(dft.index).mean()
nindex = dft.index.values
theta=np.squeeze(dft.values[:-1])
plt.figure()
# plt.plot(position.loc[:, 'ry'].as_units('ms'))
position.loc[:, 'ry'].as_units('ms').plot()
thetad = nts.Tsd(t =bins[0:-2]+ np.diff(bins[:-1])/2,d = theta, time_units='ms')
thetad.plot()
plt.plot(thetad)

dif = theta2-alphas
dif = dif[~np.isnan(dif)]
alphas = alphas + mean(dif)

uwrap = np.unwrap(alphas)
uwrap = gaussian_filter(uwrap, sigma =1) #filter
alpha_f = uwrap % (2*np.pi)

uwrap = np.unwrap(alphas)
uwrap = gaussian_filter(uwrap, sigma =1) #filter
theta = uwrap % (2*np.pi)


fig = plt.figure()
ax = plt.subplot(1,1,1)
ax.plot(theta2, color = 'tan', label = "predicted")
ax.plot(alpha_f, color = 'lightblue', label = "true")
ax.set_title ("")
ax.set_xlabel("time")
ax.set_ylabel("angle")
ax.legend()
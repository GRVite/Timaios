#!/usr/bin/env python3
# -*- coding: utf-8 -*-
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
# plt.figure(figsize=(50,60))
# for i,k in enumerate(GF.keys()):
#     plt.subplot(5,raws+1,i+1)    
#     tmp = gaussian_filter(GF[k].values, sigma = 1.3)
#     im=imshow(tmp, extent = ext, cmap = 'jet', interpolation = 'bilinear')
#     #plt.colorbar(im,fraction=0.046, pad=0.04)
#     plt.title(str([i]))
#     plt.xticks([]),plt.yticks([])
# plt.show()
# plt.suptitle('Place fields ' + session)

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
linear_sum = linear_sum.reshape(linear_sum.shape[0], grid_fields.shape[1], grid_fields.shape[2])

"""
Argmax method
"""
import cv2
#Identify the position of the highest value
filt_linear_sum = np.zeros_like(linear_sum)
xpred = []
ypred = []
for i in range(len(linear_sum)):
    filt_linear_sum[i] = gaussian_filter(linear_sum[i], sigma = 3)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(filt_linear_sum[i])
    xpred.append(max_loc[0])
    ypred.append(max_loc[1])
xpred = np.array(xpred)
ypred = np.array(ypred)



#Make dataframe 
xbins = np.linspace(ext[0], ext[1], grid_fields.shape[1])
ybins = np.linspace(ext[2], ext[3], grid_fields.shape[2])
xpred = xbins[xpred]
ypred = ybins[ypred]
pred_pos = pd.DataFrame(index = spike_counts.index, columns = ['x', 'z'], data = np.vstack((xpred, ypred)).T)


#identify the position of the animal at time t
# ti = 11500
t = int(spike_counts.index[ti]*1000)
t = position.index.values[np.argmin(np.abs(position.index.values - t))]
figure()
imshow(filt_linear_sum[ti+1], extent = ext, cmap = 'summer', origin = 'lower')
scatter([position.loc[t, 'x']], [position.loc[t, 'z']], label = 'True')
scatter([pred_pos.iloc[ti]['x']], [pred_pos.iloc[ti]['z']], color = 'red', label = 'predicted')

#Calculate theta
theta=np.arctan2(xpred,ypred)
theta = gaussian_filter(theta, sigma = 4)
fig = plt.figure()
ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
ax.plot(theta, color = 'tan')
ax.set_title ("Theta")
ax.set_xlabel("time")
ax.set_ylabel("angle")
ax.set_yticks([])
ax.set_xticks([])
# ax.set_yticklabels(["0","","","","","","360"])

"""
Bayesian
"""
from scipy.signal import correlate2d
ti = 1400

tmp = correlate2d(filt_linear_sum[ti], filt_linear_sum[ti+500])

figure()
imshow(tmp, extent = ext, cmap = 'jet', origin = 'lower')


"""
Crosscorr
"""
ti = 1200
cc = correlate2d(filt_linear_sum[ti], filt_linear_sum[ti+110],mode = 'same')

#identify highest value
min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(cc)
xpred = max_loc[0]
ypred = max_loc[1]
y1 = 0
x1 = 0
alpha = np.arctan2(ypred-filt_linear_sum.shape[2]//2, xpred-filt_linear_sum.shape[1]//2)


figure();subplot(131)
imshow(filt_linear_sum[ti].T)
plt.xticks([])
plt.yticks([])
subplot(132)
imshow(filt_linear_sum[ti+110].T)
plt.xticks([])
plt.yticks([])
subplot(133)
imshow(cc.T)
# scatter(xpred, ypred, color = 'red', label = 'predicted')
t = int(spike_counts.index[ti]*1000)
t = position.index.values[np.argmin(np.abs(position.index.values - t))]
# scatter([position.loc[t, 'x']], [position.loc[t, 'z']], label = 'True')
plt.arrow(25,25,ypred-filt_linear_sum.shape[2]//2, xpred-filt_linear_sum.shape[1]//2, width =0.3, color='red')
plt.xticks([])
plt.yticks([])
scatter([position.loc[t, 'x']], (position.loc[t, 'z']+0.25)*25, label = 'True', c="blue")  

t = int(spike_counts.index[ti]*1000)
t = position.index.values[np.argmin(np.abs(position.index.values - t))]
t2 = int(spike_counts.index[ti+110]*1000)
t2 = position.index.values[np.argmin(np.abs(position.index.values - t2))]

figure()
# imshow(filt_linear_sum[ti].T, extent = ext, cmap = 'summer', origin = 'lower')
plt.plot(position['z'].values, position['x'].values, color = "gray", alpha = 0.2)
scatter([position.loc[t, 'x']], [position.loc[t, 'z']], label = 'True', c="lightblue")  
scatter([position.loc[t2, 'x']], [position.loc[t2, 'z']], label = 'True', c="indianred")  
plt.arrow(position.loc[t, 'x'], position.loc[t, 'z'],
         position.loc[t2, 'x']-position.loc[t, 'x'], position.loc[t2, 'z']-position.loc[t, 'z'], color = "green", alpha = "0.3")
#real angle
y = position.loc[t2, 'x'] - position.loc[t, 'x']
x = position.loc[t2, 'z'] - position.loc[t, 'z']
alpha - np.arctan2(x,y)


import numpy as np
import math
import matplotlib.pyplot as plt


def plot_point(point, angle, length):
     '''
     point - Tuple (x, y)
     angle - Angle you want your end point at in degrees.
     length - Length of the line you want to plot.

     Will plot the line on a 10 x 10 plot.
     '''

     # unpack the first point
     x, y = (0,0)
     angle = alpha
     length = 1

     # find the end point
     endy = length * math.sin(math.radians(angle))
     endx = length * math.cos(math.radians(angle))

     # plot the points
     fig = plt.figure()
     ax = plt.subplot(111)
     ax.set_ylim([0, 10])   # set the bounds to be 10, 10
     ax.set_xlim([0, 10])
     ax.plot([x, endx], [y, endy])

     fig.show()


#
plt.figure()
plt.arrow(0,0, y,x)
plt.arrow(0,0, ypred-filt_linear_sum.shape[2]//2, xpred-filt_linear_sum.shape[1]//2, color='red')

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
# alphas = gaussian_filter(alphas, sigma =1) #filter
alphas [alphas<0] += 2*np.pi

# #calculate theta

px = position.loc[:, 'x'].as_units('ms').values
py = position.loc[:, 'z'].as_units('ms').values
ind = position.as_units('ms').index
index=np.digitize(ind,bins)
dfx = pd.DataFrame(index = index)
dfx['px']=px
dfx=dfx.groupby(dfx.index).median()
px=np.squeeze(dfx.values[:-1])

dfy = pd.DataFrame(index = index)
dfy['py']=py
dfy=dfy.groupby(dfy.index).median()
py=np.squeeze(dfy.values[:-1])
theta2 = np.arctan2 (px, py)
theta2[theta2<0] += 2*np.pi

#Take theta
theta = position.loc[:, 'ry'].as_units('ms').values
ind = position.as_units('ms').index
index=np.digitize(ind,bins)
dft = pd.DataFrame(index = index)
dft['theta']=theta
dft=dft.groupby(dft.index).mean()
theta=np.squeeze(dft.values[:-1])

fig = plt.figure()
ax = plt.subplot(1,1,1)
ax.plot(alphas, color = 'tan', label = "predicted")
ax.plot(theta, color = 'lightblue', label = "true")
ax.set_title ("Alpha")
ax.set_xlabel("time")
ax.set_ylabel("angle")
ax.legend()


# ax.set_yticks([])
# ax.set_xticks([])
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 29 10:56:49 2019-

@author: vite

Based on main4_raw, Starter Pack, Guillaume Viejo
"""


"""
This script will show you how to load the various data you need

The function are already written in the file wrappers.py that should be in the same directory as this script

To speed up loading of the data, a folder called /Analysis will be created and some data will be saved here
So that next time, you load the script, the wrappers will search in /Analysis to load faster
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

#
from functions import computeAngularTuningCurves
tuning_curves = computeAngularTuningCurves(spikes, position['ry'], wake_ep, 60)
from functions import smoothAngularTuningCurves
tuning_curves = smoothAngularTuningCurves(tuning_curves, 10, 2)
#Determine the number of raws
raws = round(len(spikes)/5)
plt.figure(figsize=(40,200))
for i, n in enumerate(tuning_curves.columns):
    ax=plt.subplot(5,raws+1,i+1, projection = 'polar')
    plt.plot(tuning_curves[n], color = 'darkorange')
    plt.title('Neuron' + ' ' + str(i) , loc ='center', pad=25)
plt.subplots_adjust(wspace=0.4, hspace=2, top = 0.85)
plt.show()
plt.savefig(data_directory + '/plots' + '/HD.pdf')


"""
plt.figure(figsize=(20,100))
for i, n in enumerate(tuning_curves.columns):
    plt.subplot(5,raws,i+1)
    plt.polar(tuning_curves[n], color = 'darkorange')
    plt.title('Neuron' + ' ' + str(i) , loc ='center', pad=2)
    plt.xticks(['N', '', 'W', '', 'S', 'E', ''])
plt.subplots_adjust(wspace=0.4, hspace=1, top = 1.3)
plt.show()
"""

"""
GENERAL
"""

###
#Place fields
###
GF, ext = computePlaceFields(spikes, position[['x', 'z']], wake_ep, 30)
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
plt.savefig(data_directory + '/plots' + '/GF.png')

selection = [2,3,4,6,7,16,17, 25]
plt. figure(figsize=(50,60))
for i,k in enumerate(selection):
    plt.subplot(3,4,i+1)    
    tmp = gaussian_filter(GF[k].values, sigma = 1)
    im=imshow(tmp, extent = ext, cmap = 'jet', interpolation = 'bilinear')
    #plt.colorbar(im,fraction=0.046, pad=0.04)
    plt.title(str(k))
plt.show()
plt.savefig(data_directory + '/plots' + '/GF_selected.pdf')

###
#Autocorrelograms
###
plt.figure(figsize=(40,50))
for i,k in enumerate(GF.keys()):
    plt.subplot(5,raws+1,i+1)
    tmp = gaussian_filter(GF[k].values, sigma = 0.2)
    tmp2 = correlate2d(tmp, tmp)
    imshow(tmp2, extent = ext, cmap = 'jet', interpolation = 'bilinear')
    plt.title(str([i]))
plt.savefig(data_directory + '/plots' + '/Au.pdf')

#Test rotate
from scipy import ndimage, misc
ndimage.rotate(tmp2, 45, reshape=True)

plt.figure(figsize=(4,5))
tmp = gaussian_filter(GF[2].values, sigma = 0.35)
tmp2 = correlate2d(tmp, tmp)
imshow(tmp2, cmap = 'jet', interpolation = 'bilinear')
plt.colorbar(im,fraction=0.046, pad=0.04)
plt.title(str([i]))
plt.scatter(peaks[1:,0],peaks[1:,1], marker="*",c='black')
plt.plot([29,39])

#test identify peaks
from skimage.feature import peak_local_max
peaks = peak_local_max(tmp2, num_peaks = 7)


#Spike maps
plt.figure(figsize = (15,16))
#fig.subtitle('Spikes + Path Plot',size=30)
for i,k in enumerate(selection):
    ax=subplot(3,4,i+1) #if you have more than 20cells change the numbers in bracket to reflect that
    plt.scatter(position['x'].realign(spikes[k].restrict(wake_ep)),position['z']
                .realign(spikes[k].restrict(wake_ep)),s=5,c='steelblue',label=str(k))
    legend()
    plt.plot(position['x'].restrict(wake_ep),position['z'].restrict(wake_ep),color='lightgrey', alpha=0.5)  


"""
Place fields of the first x minutes of activity for a given neuron
"""
#Select a neuron
n=selection[0]
minutes=15
start = spikes[n].start_time()
end = spikes[n].start_time()+ minutes* 60000000
interval = nts.IntervalSet(start = start, end = end)
dic = {0:spikes[7].restrict(interval)}
nposition = position[['x', 'z']].restrict(interval)
GFu, ext1 = computePlaceFields(dic, nposition , wake_ep, 30)

plt. figure(figsize=(10,30))
ax1=subplot(2,2,1)
plt.plot(position['x'], position['z'])
plt.title("Tracking data")
ax2=subplot(2,2,2)
tmp = gaussian_filter(GF[n].values, sigma = 1.1)
im=imshow(tmp, extent = ext, cmap = 'jet', interpolation = 'bilinear')
plt.title("Neuron " + str(n))
plt.gcf().text(0.5, 0.01, str(minutes) + ' min')
plt.colorbar(im,fraction=0.046, pad=0.04)
ax2.invert_yaxis()  
            
ax3=subplot(2,2,3)
plt.plot(nposition['x'], nposition['z'])
plt.title("Tracking data")

ax4=subplot(2,2,4)
tmp = gaussian_filter(GFu[0].values, sigma = 1.1)
im=imshow(tmp, extent = ext1, cmap = 'jet', interpolation = 'bilinear')
plt.title("Neuron " + str(n))
plt.gcf().text(0.5, 0.01, str(minutes) + ' min')
plt.colorbar(im,fraction=0.046, pad=0.04)
ax4.invert_yaxis()

"""
Place fields 
"""

#First half
duration = spikes[n].end_time() - spikes[n].start_time()
start = spikes[n].start_time()
end = spikes[n].start_time()+ duration/2
interval = nts.IntervalSet(start = start, end = end)
dic = {0:spikes[7].restrict(interval)}
nposition1 = position[['x', 'z']].restrict(interval)
GFu1, ext1 = computePlaceFields(dic, nposition1 , wake_ep, 30)

#Second half
start = spikes[n].start_time()+ duration/2
end = spikes[n].end_time()
interval = nts.IntervalSet(start = start, end = end)
dic = {0:spikes[7].restrict(interval)}
nposition2 = position[['x', 'z']].restrict(interval)
GFu2, ext2 = computePlaceFields(dic, nposition2 , wake_ep, 30)

#Plots
plt. figure(figsize=(8,40))
ax1=subplot(3,2,1)
plt.plot(position['x'], position['z'])
plt.title(str(int(duration/1000/1000/60)) + " min")
ax2=subplot(3,2,2)
tmp = gaussian_filter(GF[n].values, sigma = 1.1)
im=imshow(tmp, extent = ext, cmap = 'jet', interpolation = 'bilinear')
plt.title("Complete")
plt.colorbar(im,fraction=0.046, pad=0.04)
ax2.invert_yaxis()  

ax3=subplot(3,2,3)
plt.plot(nposition1['x'], nposition1['z'])
plt.title("0 - " + str(int(duration/1000/1000/60/2)) + " min")
ax4=subplot(3,2,4)
tmp = gaussian_filter(GFu1[0].values, sigma = 1.1)
im=imshow(tmp, extent = ext1, cmap = 'jet', interpolation = 'bilinear')
plt.title("First half")
plt.colorbar(im,fraction=0.046, pad=0.04)
ax4.invert_yaxis()  

ax5=subplot(3,2,5)
plt.plot(nposition2['x'], nposition2['z'])
plt.title(str(int(duration/1000/1000/60/2)) + " - " +str(int(duration/1000/1000/60)) + " min")
ax6=subplot(3,2,6)
tmp = gaussian_filter(GFu2[0].values, sigma = 1.1)
im=imshow(tmp, extent = ext2, cmap = 'jet', interpolation = 'bilinear')
plt.title("Second half")
plt.colorbar(im,fraction=0.046, pad=0.04)
ax6.invert_yaxis()  

plt.tight_layout()
plt.suptitle("Comparisson ", x = 0.5, y = 1)
plt.savefig(data_directory + '/plots' + '/Comparisson.pdf')



"""
Summary figure for one neuron 
"""
#Select a neuron
n=selection[0]
plt. figure(figsize=(10,30))

ax1=subplot(2,2,1)
ax1.set_axis_off()
plt.plot(position['x'], position['z'], c='steelblue',)
plt.title("A", loc ='left',fontsize=25)
plt.box(False)

ax2=subplot(2,2,2)
ax2.invert_yaxis()  
ax2.set_axis_off()
tmp = gaussian_filter(GF[n].values, sigma = 1.1)
im=imshow(tmp, extent = ext, cmap = 'jet', interpolation = 'bilinear')
plt.title("B", loc ='left',fontsize=25)
# plt.colorbar(im,fraction=0.046, pad=0.04)
plt.box(False)
            
ax3=subplot(2,2,3)
ax3.set_axis_off()
plt.scatter(position['x'].realign(spikes[2].restrict(wake_ep)),position['z'].
            realign(spikes[2].restrict(wake_ep)), s=5, c='steelblue', label=str(n))
plt.plot(position['x'].restrict(wake_ep),position['z'].restrict(wake_ep),color='lightgrey', alpha=0.5)  
plt.title("C", loc ='left',fontsize=25)
plt.box(False)

ax4=subplot(2,2,4)
tmp = gaussian_filter(GF[n].values, sigma = 0.2)
tmp2 = correlate2d(tmp, tmp)
imshow(tmp2, extent = ext, cmap = 'jet', interpolation = 'bilinear')
ax4.set_axis_off()
plt.box(False)
plt.title("D", loc ='left',fontsize=25)

plt.savefig(data_directory + '/plots' + '/figure_proposal_n' + str(n) + '.pdf')

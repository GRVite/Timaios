#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 10 22:19:40 2020

@author: vite
"""

import os
import numpy as np
import pandas as pd
import neuroseries as nts
#from pylab import *
#import matplotlib.pyplot as plt
from functions import *
from wrappers import *
from functions import computeAngularTuningCurves

#Load data
#rootDir = '/media/3TBHDD/Data'
rootDir = '/Users/vite/navigation_system/Data'
ID = 'A4405'
session = 'A4405-200312'
wakepos=0
events = ['0'] 
data_directory = rootDir + '/' + ID + '/' + session + '/' + session
if os.path.exists(data_directory+'/plots')==False:
    os.mkdir(data_directory+'/plots')
path =rootDir + '/' + ID + '/' + session
#count number of sessions
ns = int([i for i in os.listdir(path)  if os.path.isdir(path+'/'+i)==True][0][-1:])
episodes = ['wake' if i==wakepos else 'sleep' for i in list(range(ns))]

spikes, shank = loadSpikeData(data_directory)
n_channels, fs, shank_to_channel = loadXML(data_directory)
position = loadPosition(data_directory, events, episodes, n_ttl_channels = 2, optitrack_ch = 0)
wake_ep                             = loadEpoch(data_directory, 'wake', episodes)
sleep_ep                             = loadEpoch(data_directory, 'sleep')                    
ttl_track, ttl_opto_start, ttl_opto_end = loadTTLPulse2(os.path.join(data_directory, session+'_0_analogin.dat'), 2)
ttl_track = nts.Ts(ttl_track.index.values, time_units = 's')
ttl_opto_start = nts.Ts(ttl_opto_start.index.values, time_units = 's')
ttl_opto_end = nts.Ts(ttl_opto_end.index.values, time_units = 's')
opto_ep = nts.IntervalSet(start = ttl_opto_start.index.values, end = ttl_opto_end.index.values)

#Parameters
stim = 5000
delay = 20000 #in us
duration = 1 #in min
sfreq = ((duration*1000*1000)/(stim+delay))
intensities = ['low','med','high']

#Find the beginning and end of the stimulation
t, _ = scipy.signal.find_peaks(np.diff(ttl_opto_start.as_units('s').index), height = 1)
stim_ep = np.sort(np.hstack(([ttl_opto_start.index[0]], ttl_opto_start.index[t], ttl_opto_start.index[t+1], ttl_opto_end.index[-1])))
stim_ep = stim_ep.reshape(len(stim_ep)//2, 2)
stim_ep = nts.IntervalSet(start = stim_ep[:,0], end = stim_ep[:,1])
figure()
plot(ttl_opto_start.index)
[axhline(stim_ep.loc[i,'start']) for i in stim_ep.index]
[axhline(stim_ep.loc[i,'end']) for i in stim_ep.index]
plot(ttl_opto_start.index, 'o')
show()

high_ep=stim_ep.loc[[0]]
#Take the first 10 periods of stimulation
spikes_a=np.asarray(spikes_list)
pd.DataFrame(columns=spikes.keys())

neuron = i
spikes_list = []
span=20000
for i in ttl_opto_start.restrict(high_ep).index.values:
    interval = nts.IntervalSet(start=i - span , end=i+span)
    print(interval)
    t = spikes[neuron].restrict(interval).index.values - i
    spikes_list.append(t)
    
lineSize=0.5
left, bottom, width, height = (0, 0, 5000, len(ttl_opto_start.restrict(high_ep)))
rect = plt.Rectangle((left, bottom), width, height, facecolor="limegreen", alpha=0.1)
fig, ax = plt.subplots()
ax.add_patch(rect)
ax.eventplot(spikes_list, linelengths = 30, color='black')
ax.set_ylabel('Trials')
ax.set_xlabel('Time (us)')
ax.legend(["Period of stimulation"])
ax.set_title("Raster plot for all the trials")
 
#Tuning curves from baseline activity
wake_base = nts.IntervalSet(start = wake_ep.loc[0,'start'], end=wake_ep.loc[0,'start']+ttl_opto_start.index.values[0]-1000000)
tuning_curves_base = computeAngularTuningCurves(spikes, position['ry'], wake_base, 60)
tuning_curves_base = smoothAngularTuningCurves(tuning_curves_base, 10, 2)

#Low intensity 
wake_firststim = nts.IntervalSet(start = ttl_opto_start.index[0], end=ttl_opto_start.index[0]+1*60*1000*1000)
tuning_curves_stim = computeAngularTuningCurves(spikes, position['ry'], wake_firststim, 60)
tuning_curves_stim = smoothAngularTuningCurves(tuning_curves_stim, 10, 2)

#A. Tuning curves of control vs stimulation
#Subplot
lista=["Control","Stimulation"]
plt.figure(figsize=[20,10])
for i, n in enumerate (spikes.keys()):
    ax = plt.subplot(2,5,i+1, projection='polar')
    ax.plot(tuning_curves_base[n], color ='black')
    #ax.fill(tuning_curves_base[n],"black", alpha = 0.15) 
    ax.plot(tuning_curves_stim[n], color ='lime')
    #ax.fill(tuning_curves_stim[n],"lime", alpha = 0.15) 
    ax.set_title("Neuron_" + str(n))
#    ax.legend(lista)
plt.tight_layout()
plt.savefig(data_directory + '/plots' + '/tun_baseVSstim_tot'  + '.pdf', bbox_inches = 'tight')

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 24 21:47:16 2020

@author: vite
"""

import os
import numpy as np
import pandas as pd
import neuroseries as nts
#from pylab import *
import matplotlib.pyplot as plt
import seaborn as sns
from functions import *
from wrappers import *
from functions import *
from scipy.stats import pearsonr
from vitemod import *
from vitemod import timaios as ts

"""
Load data
"""
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
ns = int([i for i in os.listdir(path) if os.path.isdir(path+'/'+i)==True][-1][-1:])
if ns == 1:
    episodes=['wake']
else:
    episodes = ['wake' if i==wakepos else 'sleep' for i in list(range(ns+1))]
spikes, shank = loadSpikeData(data_directory)
n_channels, fs, shank_to_channel = loadXML(data_directory)
position = loadPosition(data_directory, events, episodes, n_ttl_channels = 2, optitrack_ch = 0)
wake_ep                             = loadEpoch(data_directory, 'wake', episodes)
if "sleep" in episodes:
    sleep_ep                             = loadEpoch(data_directory, 'sleep')                    
ttl_track, ttl_opto_start, ttl_opto_end = loadTTLPulse2(os.path.join(data_directory, session+'_0_analogin.dat'), 2)
ttl_track = nts.Ts(ttl_track.index.values, time_units = 's')
ttl_opto_start = nts.Ts(ttl_opto_start.index.values, time_units = 's')
ttl_opto_end = nts.Ts(ttl_opto_end.index.values, time_units = 's')
opto_ep = nts.IntervalSet(start = ttl_opto_start.index.values, end = ttl_opto_end.index.values)
stim_ep=manage.optoeps(ttl_opto_start, ttl_opto_end) #Load main stim epochs


neuron = 5
# whole duration= int(spikes[neuron].as_units('s').index[-1] - spikes[21].as_units('s').index[0])
bins = np.arange(0, 300000, 1)
plt.figure()
plt.hist(np.diff(spikes[neuron].index.values), bins)



bins = np.arange(0, 300000+10000, 10000)
units = 'us'
neurons = []
isi = []
shanks = []

plt.figure()
for n in spikes.keys():
    diff = np.diff(spikes[n].as_units(units).index.values)
    ax = plt.subplot(5,5,n+1)
    x = ax.hist(diff, bins)
    isi.append(diff)
    for i in range(len(diff)):
        neurons.append(n)
        shanks.append(shank[n])
plt.tight_layout()
isi = np.concatenate(isi).ravel()

df = pd.DataFrame(data = np.stack ((isi, neurons, shanks), axis = 1), columns = ["times", "neurons", "shanks"])

fog, axs = plt.subplots(5,5)
for i in neurons:
    data = df["times"][df["neurons"]==i]
    sns.distplot(data, ax = axs[i//5, i%3])
plt.show()

#spikes per trial
trials = len(data_by_trials)
frate_trial = []
for t in trials:
    frate_trial.append (sum(neuron[t])/time)
mean_reponse = mean(frate_trial)

noise = []
for i in len(frate_trial):
    noise.append(frate_trial[i])

variance = np.var(firingRate)


#test
neuron = data_by_trials[5,:,:]
print((data_by_trials[1,:,:]).shape)
plt.figure()
plt.plot(neuron)
trials = range(neuron.shape[0])
frate_trial = []
for t in trials:
    frate_trial.append (sum(neuron[t])/time)
mean_reponse = mean(frate_trial)

noise = []
for i in len(frate_trial):
    noise.append(frate_trial[i])
    
    
#Calculate noise
neuron = data_by_trials[5,:,:]
noise = calcnoise(neuron, len_trials)
print(noise)


# Calculate noise for a given population of neurons

def noise_pop (data_by_trials, indexes, len_trials):
    noise = []
    for i in indexes:
        neuron = data_by_trials[i,:,:]
        print(neuron)
        noise_n = calcnoise(neuron, len_trials)
        # Sum the noise present in all trials, for one neuron
        sum_trial = np.sum(noise_n)
        print(sum_trial)
        noise.append(sum_trial)
    return noise

# select some neurons
indexes = [5,6,7]
# calculate the noise present in all neurons
noisePopulation = noise_pop (data_by_trials, indexes, len_trials)
# compute the variance of this noise
print(np.var(noisePopulation))


#Calculate noise raw
neuron = data_by_trials[5,:,:]
trials = range(neuron.shape[0])
frate_trial = []
for t in trials:
    frate_trial.append (sum(neuron[t])/len_trials[t])
mean_reponse = np.mean(frate_trial)
noise = []
for i in trials:
    noise.append(frate_trial[i])

neurons = range(neuron.shape[])

def noise_pop (data, indexes, len_trials)
    noise = []
    for i in indexes:
        noise.append(calcnoise(data[i,;,:], len_trials)
    return np.sum(noise)
indexes = [1,2,3]
noise_pop (data_by_trials, indexes, len_trials)





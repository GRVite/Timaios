#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 26 16:15:05 2019

@author: grvite
"""

import os
import numpy as np
import pandas as pd
import neuroseries as nts
#from pylab import *
import matplotlib.pyplot as plt
from wrappers import loadEpoch
import os
from wrappers import loadSpikeData
from wrappers import loadXML
from wrappers import loadPosition
from functions import *
from functions import makeRingManifold


#def makeRingManifold(spikes, ep, bin_size = 200):    
#    """
#    spikes : dict of hd spikes
#    ep : epoch to restrict    
#    bin_size : in ms
#    """
#    neurons = np.sort(list(spikes.keys()))
#    bins = np.arange(ep.as_units('ms').start.iloc[0], ep.as_units('ms').end.iloc[-1]+bin_size, bin_size)
#    spike_counts = pd.DataFrame(index = bins[0:-1]+np.diff(bins)/2, columns = neurons)
#    for i in neurons:
#        spks = spikes[i].as_units('ms').index.values
#        spike_counts[i], _ = np.histogram(spks, bins)
#    
#    rates = np.sqrt(spike_counts/(bin_size))
#    tmp=rates.rolling(window=200,win_type='gaussian',center=True,min_periods=1).mean(std=5)
#    tmp = tmp.values
#    imap = Isomap(n_neighbors = 100, n_components = 2, n_jobs = -1).fit_transform(tmp)
#    iwak = imap
#    return iwak


rootDir = '/media/3TBHDD/Data'
ID = 'A3302'
session = 'A3302-190821'
data_directory = rootDir + '/' + ID + '/' + session + '/' + session

files = os.listdir(data_directory) 

spikes, shank = loadSpikeData(data_directory)

n_channels, fs, shank_to_channel = loadXML(data_directory)

episodes = ['wake']
events = ['0']

wake_ep = loadEpoch(data_directory, 'wake', episodes)


position = loadPosition(data_directory, events, episodes, n_ttl_channels = 1, optitrack_ch = 0)



tuning_curves = computeAngularTuningCurves(spikes, position['ry'], wake_ep, 60)
tuning_curves = smoothAngularTuningCurves(tuning_curves, 10, 2)

tokeep, stat = findHDCells(tuning_curves)

#figure()
#for i in range(len(tokeep)):
#    subplot(3,3,i+1, projection = 'polar')
#    plot(tuning_curves[tokeep[i]])

spikes = {n:spikes[n] for n in tokeep}

ep = nts.IntervalSet(start = wake_ep.loc[0,'start'], end = wake_ep.loc[0,'start']+30*60*1000*1000)



iwak = makeRingManifold(spikes, ep, position['ry'], position[['x', 'z']], bin_size = 300)


figure()
for i, n in enumerate(spikes):
    subplot(5,4,i+1, projection = 'polar')
    plot(tuning_curves[n])

show()



















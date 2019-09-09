#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 26 17:35:40 2019

@author: grvite
"""

from sklearn.manifold import Isomap
import numpy as np
import pandas as pd
import neuroseries as nts
from pylab import *
from mpl_toolkits.mplot3d import Axes3D

def makeRingManifold(spikes, ep, bin_size = 200):    
    """
    spikes : dict of hd spikes
    ep : epoch to restrict    
    bin_size : in ms
    """
    neurons = np.sort(list(spikes.keys()))
    bins = np.arange(ep.as_units('ms').start.iloc[0], ep.as_units('ms').end.iloc[-1]+bin_size, bin_size)
    spike_counts = pd.DataFrame(index = bins[0:-1]+np.diff(bins)/2, columns = neurons)
    for i in neurons:
        spks = spikes[i].as_units('ms').index.values
        spike_counts[i], _ = np.histogram(spks, bins)
    
    rates = np.sqrt(spike_counts/(bin_size))
    tmp=rates.rolling(window=200,win_type='gaussian',center=True,min_periods=1).mean(std=5)
    tmp = tmp.values
    imap = Isomap(n_neighbors = 100, n_components = 3, n_jobs = -1).fit_transform(tmp)
    iwak = imap
    return iwak

data_directory = '/media/3TBHDD/Data/A3302/A3302-190822/A3302-190822'

from wrappers import loadSpikeData
from wrappers import loadXML
from wrappers import loadPosition
from wrappers import loadEpoch

spikes, shank = loadSpikeData(data_directory)
n_channels, fs, shank_to_channel = loadXML(data_directory)
episodes = ['wake']
events = ['0']


position = loadPosition(data_directory, events, episodes, n_ttl_channels = 1, optitrack_ch = 0)
wake_ep = loadEpoch(data_directory, 'wake', episodes)

tokeep = [6, 8, 9, 17, 20, 23, 24, 25, 28, 30, 37]

spikes = {n:spikes[n] for n in tokeep}

t = wake_ep.loc[0,'start']+15*60*1000*1000
cutwake_ep = nts.IntervalSet(start = t, end = t + 30*60*1000*1000)

iwak = makeRingManifold(spikes, cutwake_ep)


figure()
ax = subplot(111, projection = '3d')    
ax.set_aspect(aspect=1)
ax.scatter(iwak[:,0], iwak[:,1], iwak[:,2], alpha = 0.5, linewidth = 0)
show()

















#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 13 15:30:45 2020

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
from functions import computeAngularTuningCurves
from scipy.stats import pearsonr

#Load data
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
spikes, shank = loadSpikeData(data_directory)
n_channels, fs, shank_to_channel = loadXML(data_directory)
position = loadPosition(data_directory, events, episodes, n_ttl_channels = 2, optitrack_ch = 0)
wake_ep                             = loadEpoch(data_directory, 'wake', episodes)

position = loadPosition(data_directory, events, episodes, n_ttl_channels = 1, optitrack_ch = 0)
SpeedTC= computeSpeedTuningCurves(spikes, position, wake_ep)

SpeedTC.plot()

selection = []
for i in spikes.keys():
    if SpeedTC[i].min() > 10:
        selection.append(i)
        
plt.figure()
for i in selection:
    SpeedTC[i].plot()
plt.xlabel("speed")
plt.ylabel("firing rate (Hz)")
plt.show()
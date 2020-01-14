#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 20 15:13:40 2019

@author: grvite
"""

import os
import numpy as np
import pandas as pd
import neuroseries as nts
#from pylab import *
import matplotlib.pyplot as plt
from functions import *
from wrappers import *
from functions import *

rootDir = '/media/3TBHDD/Data'
ID = 'A4203'
session = 'A4203-191220'
data_directory = rootDir + '/' + ID + '/' + session + '/' + session

if os.path.exists(data_directory+'/plots') ==False:
    os.mkdir(data_directory+'/plots')

episodes = ['wake', 'wake', 'sleep']
events = ['0', '1']

spikes, shank = loadSpikeData(data_directory)
n_channels, fs, shank_to_channel = loadXML(data_directory)
position = loadPosition(data_directory, events, episodes, n_ttl_channels = 2, optitrack_ch = 0)
wake_ep                             = loadEpoch(data_directory, 'wake', episodes)
sleep_ep                             = loadEpoch(data_directory, 'sleep')                    

ttl_track, ttl_opto = loadTTLPulse2(os.path.join(data_directory, session+'_1_analogin.dat'), 2)
ttl_opto = nts.Ts(ttl_opto.index.values, time_units = 's')

figure()
plt.plot(ttl_opto.index.values, 'o')
plt.show()

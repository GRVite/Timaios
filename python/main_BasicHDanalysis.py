#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 19 15:13:42 2020

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
import pickle
from wrappers import *

"""
A. LOAD DATA
"""

# def analysis(data_directory_load, dir2save_plots, ID, session):
data_directory_load = '/Volumes/LaCie/Timaios/Kilosorted/A4404/A4404-200606/my_data/'
    
# load data
spikes = pickle.load(open(data_directory_load + '/spikes.pickle', 'rb'))
shank = pickle.load(open(data_directory_load  + '/shank.pickle', 'rb'))
episodes = pickle.load(open(data_directory_load + '/episodes.pickle', 'rb'))
position = pickle.load(open(data_directory_load  + '/position.pickle', 'rb'))
wake_ep = pickle.load(open(data_directory_load  + '/wake_ep.pickle', 'rb'))
opto_ep = pickle.load(open(data_directory_load  + '/opto_ep.pickle', 'rb'))
stim_ep = pickle.load(open(data_directory_load  + '/stim_ep.pickle', 'rb'))

dir2save_plots = data_directory_load + 'plots'

"""
All, whole recording
"""
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
All, selected interval
"""

interval = nts.IntervalSet(start = wake_ep.loc[0,'start'], 
                           end=stim_ep.loc[0, 'start'] -1000)
tuning_curves = computeAngularTuningCurves(spikes, position['ry'], interval, 60)
tuning_curves = smoothAngularTuningCurves(tuning_curves, window = 10, deviation = 2)

#Determine the number of raws
#raws = round(len(spikes)/5)
raws = 3
plt.figure(figsize=(40,200))
for i, n in enumerate(tuning_curves.columns):
    ax=plt.subplot(1, raws, i+1, projection = 'polar')
    plt.plot(tuning_curves[n], color = 'darkorange')
    plt.title('Neuron' + ' ' + str(i) , loc ='center', pad=25)
plt.subplots_adjust(wspace=0.4, hspace=2, top = 0.85)
plt.show()
plt.savefig(dir2save_plots + '/HD.pdf')

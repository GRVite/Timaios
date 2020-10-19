#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 30 12:47:10 2020

@author: mamiferock
"""

"""
    This scripts extracts the information from the original data,
    and export it to pickle format. This allows to have a very light folder 
    that is easy to share. 
"""

from functions import *
from my_functions import *
from wrappers import *
import pandas as pd
import pickle
import os

#This allows you to read a google sheet with the information of the sessions
#recorded.
SCOPES = ['https://www.googleapis.com/auth/spreadsheets']
SAMPLE_SPREADSHEET_ID_input = '1DiJMx6G9IhU_X_QY6NTWbqBWh5CvvLsoQVdo4IN0KXc'
SAMPLE_RANGE_NAME = 'A1:AA100'
values = accessgoogle(SAMPLE_SPREADSHEET_ID_input, SAMPLE_RANGE_NAME)
df = pd.DataFrame(values[1:], columns=values[0])
print(df)


# rootDir = '/Users/vite/navigation_system/Data'
rootDir = '/Volumes/LaCie/Timaios/Kilosorted'
# Select animal ID and session
ID = 'A4403'
session = 'A4403-200626'

# Load the spikes and shank data
data_directory =  rootDir + '/' + ID + '/' + session +  '/' + session
data_directory = '/Volumes/LaCiel/Timaios/Kilosorted/A4403/A4403-200626/A4403-200626'
spikes, shank = loadSpikeData(data_directory)

# Find the number of events
events = []
for i in os.listdir(data_directory):
    if os.path.splitext(i)[1]=='.csv':
        if i.split('_')[1][0] != 'T': 
            events.append( i.split('_')[1][0])
# Get a list with the episodes for the given session
episodes = df[df['Session']==session]["Episodes"].values[0].split(',')

# Load the position of the animal derived from the camera tracking
position = loadPosition(data_directory, events, episodes, n_ttl_channels = 2, optitrack_ch = 0)

# Get the time interval of the wake epoch
wake_ep = loadEpoch(data_directory, 'wake', episodes)
if 'sleep' in episodes:
    sleep_ep = loadEpoch(data_directory, 'sleep')
    with open(data_directory + '/my_data/' + 'sleep_ep' + '.pickle', 'wb') as handle:
        pickle.dump(sleep_ep, handle, protocol=pickle.HIGHEST_PROTOCOL)

# Get a pandas Series with the times of the TTL pulses aligned with the optogenetic stimulation                 
ttl_track, ttl_opto_start, ttl_opto_end = loadTTLPulse2(os.path.join(
    data_directory, session + '_' + events[0] + '_' + 'analogin.dat'), 2)
# Transform the pandas Series into a time series 
ttl_opto_start = nts.Ts(ttl_opto_start.index.values, time_units = 's')
ttl_opto_end = nts.Ts(ttl_opto_end.index.values, time_units = 's')
# Create a time interval corresponding to the optogenetic stimulation epoch
opto_ep = nts.IntervalSet(start = ttl_opto_start.index.values, end = ttl_opto_end.index.values)
# Get the start and end times of the three diferent light intensities of the stimulation
stim_ep = optoeps(ttl_opto_start, ttl_opto_end) 

# position.index[0], stim_ep.loc[0].start, stim_ep.loc[1].start, stim_ep.loc[2].start,  
# stim_ep.loc[0].start - 400, stim_ep.loc[1].end, stim_ep.loc[2], position[-1]
# start = []
# end = []
# for i in range(stim_ep.index):
#     start.append(stim_ep.loc[i].start)
#     end.append(stim_ep.loc[i].end)
#     start.append (stim_ep.loc[i+1].end + 400)
#     end.append (stim_ep.loc[i+2].start - 400)


#create a new directory for saving the data
os.mkdir(data_directory + '/my_data')
#save data in pickle format
# for string, objct in zip(['spikes', 'shank', 'episodes', 'position', \
#               'wake_ep', 'opto_ep', 'stim_ep'],
#              [spikes, shank, episodes, position, wake_ep, opto_ep, stim_ep]):

for string, objct in zip(['spikes', 'shank', 'episodes', 'position', \
              'wake_ep'],
             [spikes, shank, episodes, position, wake_ep]):
    with open(data_directory + '/my_data/' + string + '.pickle', 'wb') as handle:
        pickle.dump(objct, handle, protocol=pickle.HIGHEST_PROTOCOL)
   
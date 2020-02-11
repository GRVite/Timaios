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
#import matplotlib.pyplot as plt
from functions import *
from wrappers import *
from functions import *
from functions import computeAngularTuningCurves

#Load data
#rootDir = '/media/3TBHDD/Data'
rootDir = '/Users/vite/navigation_system/Data'
ID = 'A4203'
session = 'A4203-191221'
wakepos=0
events = ['0'] 
data_directory = rootDir + '/' + ID + '/' + session + '/' + session
if os.path.exists(data_directory+'/plots')==False:
    os.mkdir(data_directory+'/plots')
path =rootDir + '/' + ID + '/' + session
#count number of sessions
ns = int([i for i in os.listdir(path)  if os.path.isdir(path+'/'+i)==True][-1][-1:])
episodes = ['wake' if i==wakepos else 'sleep' for i in list(range(ns+1))]

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


### mio
wake_base = nts.IntervalSet(start = wake_ep.loc[0,'start'], end=wake_ep.loc[0,'start']+ttl_opto_start.index.values[0]-1)
tuning_curves_base = computeAngularTuningCurves(spikes, position['ry'], wake_base, 60)
tuning_curves_base = smoothAngularTuningCurves(tuning_curves_base, 10, 2)

#wake_firststim = nts.IntervalSet(start = wake_ep.loc[0,'start']+ttl_opto_start.index[0], end=wake_ep.loc[0,'start']+ttl_opto_start.index[999])
wake_firststim = opto_ep
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
    

#Individual


lista=["Control","Stimulation"]
for n in spikes.keys():
    figp = plt.figure(figsize=(8,8))
    ax = figp.add_subplot(111, projection='polar')
    ax.plot(tuning_curves_base[n], color ='black')
    #ax.fill(tuning_curves_base[n],"black", alpha = 0.15) 
    ax.plot(tuning_curves_stim[n], color ='lime')
    #ax.fill(tuning_curves_stim[n],"lime", alpha = 0.15) 
    ax.set_title("Neuron_" + str(n))
    ax.legend(lista)
    plt.savefig(data_directory + '/plots' + '/tun_baseVSstim_' + str(n) + '.pdf', bbox_inches = 'tight')
    
#Get the time intervals of the stimulation where the neuron was firing in its preferred direction
angle = position['ry'].realign(ttl_opto_start)
angle= pd.DataFrame(data=angle.values, index=angle.index.values, columns=['angle'])
limit_inf=(2*np.pi/8)*6
limit_sup=(2*np.pi/8)*7
angle['label']=(angle['angle'] >= limit_inf) & (angle['angle']<= limit_sup)
angle=angle[angle['label']]

#Use these intervals to restrict the time of the spikes
neuron = 7
spikes_list = []
for i in range(len(angle.index)):
    print(i)
    interval = nts.IntervalSet(start=angle.index[i] - 15000 , end=angle.index[i]+ 25000)
    t = spikes[neuron].restrict(interval).index.values - angle.index[i]
    spikes_list.append(t)
lineSize=0.5
plt.eventplot(spikes_list, linelengths = 30)

#Firing rate
MFirRate1 = computeMeanFiringRate(spikes, [wake_base],["base"])
MFirRate2 = computeMeanFiringRate(spikes, [wake_firststim],["stim"])
MFirRate = pd.concat ([MFirRate1, MFirRate2], axis=1)




"""
In progress...
"""

#Find the beginning and end of the stimulation
t, _ = scipy.signal.find_peaks(np.diff(ttl_opto_start.as_units('s').index), height = 1)
stim_ep = np.sort(np.hstack(([ttl_opto_start.index[0]], ttl_opto_start.index[t], ttl_opto_start.index[t+1], ttl_opto_end.index[-1])))
stim_ep = stim_ep.reshape(len(stim_ep)//2, 2)
stim_ep = nts.IntervalSet(start = stim_ep[:,0], end = stim_ep[:,1])
figure()
plot(ttl_opto_start.index)
[axhline(stim_ep.loc[i,'start']) for i in stim_ep.index]
[axhline(stim_ep.loc[i,'end']) for i in stim_ep.index]
plot(ttl_opto_start.index)
show()
#Take the first 10 periods of stimulation
stim_ep.loc[0:9]
neuron = 7
spikes_list = []
for i in range(10):
    print(i)
    interval = nts.IntervalSet(start=stim_ep['start'][i] - 2000000 , end=stim_ep['end'][i]+2000000)
    print(interval)
    t = spikes[neuron].restrict(interval).index.values - stim_ep['start'][i]
    spikes_list.append(t)
    
    
fig, (ax1,ax2) = plt.subplots(2, 1, sharex=True)
ax1.eventplot(spikes[neuron].restrict(interval).as_units('s').index)
ax2.hist(spikes[neuron].restrict(interval).as_units('s').index, bins = 10)
plt.show()



opto_stim=nts.IntervalSet(start = opto_ep.loc[0,'start'], end=opto_ep.iloc[-1,1])

wake_2ndststim = opto_ep.iloc[1000:2000]
interval = nts.IntervalSet(start=stim_ep['start'][1], end=stim_ep['end'][1])
plt.figure()
plt.eventplot(spikes[7].restrict(interval).index)
plt.show()

plt.figure()
plt.eventplot(spikes[7].restrict(opto_stim).index)
plt.show()

fig, (ax1,ax2) = plt.subplots(2, 1, sharex=True)
ax1.eventplot(spikes[neuron].restrict(wake_base).index)
ax2.hist(spikes[neuron].restrict(wake_base).index, bins = 20)
plt.show()






### mio
plt.figure()
plt.plot(ttl_track)
plt.show()
plt.plot(ttl_opto)
figure()
plt.plot(ttl_opto.index.values, 'o')
plt.show()

#Beginning of opto stimulation
ttl_opto.index.values[0]/1000/1000
ttl_track.index.values[-1]/1000/1000
ttl_opto.as_units('s').index.values[-1]


#Histogram
bin_size = 2000 # microseconds
tstart = 15000
tend = 25000
bins = np.arange(-tstart, tend+bin_size, bin_size)
timestep = bins[0:-1] + np.diff(bins)/2
rasters = {}
counts = {}
countspd= pd.Series(index = timestep)
df_tuning.to_hdf(hdf_dir + '/df_tuning.hdf', 'tuning')   


    
#Guillaume's
for i in spikes:
    #rasters[i] = []
    count = []
    for j,t in enumerate(ttl_opto_start.index.values):
        ep = nts.IntervalSet(start = t-tstart, end = t+tend)        
        tmp = spikes[i].restrict(ep)        
        if len(tmp):            
            count.append(np.histogram(tmp.index.values, bins+t)[0])            
            #rasters[i].append(count)
    #rasters[i] = np.array(rasters[i])
    counts[i] = pd.Series(index = timestep, data = np.sum(np.array(count), 0))


    
#End Histogram

from pylab import *
figure()
ax = subplot(211)
tmp = (rasters[0]*0+1).sum(1)
bin_size = 100000
bins = np.arange(-2*1e6, 3*1e6+bin_size, bin_size)
tmp = tmp.groupby(np.digitize(tmp.index.values, bins)).sum()
count = pd.Series(index = bins[0:-1] + np.diff(bins)/2, data = 0)
count.iloc[tmp.index.values-1] = tmp.values
ax2 = subplot(212, sharex = ax)
plot(rasters[0], 'k.')

plt.figure(figsize=(40,200))
for i, n in enumerate(tuning_curves.columns):
    ax=plt.subplot(5,5+1,i+1, projection = 'polar')
    plt.plot(tuning_curves[n], color = 'darkorange')
    plt.title('Neuron' + ' ' + str(i) , loc ='center', pad=25)
plt.subplots_adjust(wspace=0.4, hspace=2, top = 0.85)
plt.show()
plt.savefig(data_directory + '/plots' + '/HD.pdf')



#hell
#for one
rasters = {}
for j,t in enumerate(ttl_opto.index.values):
    print(j)
    tmp = spikes[1].loc[t-2*1e6:t+3*1e6]
    tmp.index -= t
    if len(tmp):
        rasters[j] = tmp.fillna(j)
    

#rasters[i] = pd.DataFrame.from_dict(rasters[i])

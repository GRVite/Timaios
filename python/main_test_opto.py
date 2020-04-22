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
import seaborn as sns
from functions import *
from wrappers import *
from functions import *
from functions import computeAngularTuningCurves
from scipy.stats import pearsonr

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



### mio
wake_base = nts.IntervalSet(start = wake_ep.loc[0,'start'], end=opto_ep.loc[0,'start']-10000)
tuning_curves_base = computeAngularTuningCurves(spikes, position['ry'], wake_base, 60)
tuning_curves_base = smoothAngularTuningCurves(tuning_curves_base, 10, 2)


#wake_firststim = nts.IntervalSet(start = wake_ep.loc[0,'start']+ttl_opto_start.index[0], end=wake_ep.loc[0,'start']+ttl_opto_start.index[999])
wake_firststim = opto_ep
tuning_curves_stim = computeAngularTuningCurves(spikes, position['ry'], wake_firststim, 60)
tuning_curves_stim = smoothAngularTuningCurves(tuning_curves_stim, 10, 2)

"""
A. Tuning curves of control vs stimulation
"""
#Subplot

plt.figure(figsize=[20,80])
cols = 4
raws=round(len(spikes)/cols)
for i, n in enumerate (spikes.keys()):
    ax = plt.subplot(raws,cols,i+1, projection='polar')
    ax.plot(tuning_curves_base[n], color ='black')
    #ax.fill(tuning_curves_base[n],"black", alpha = 0.15) 
    ax.plot(tuning_curves_stim[n], color ='lime')
    #ax.fill(tuning_curves_stim[n],"lime", alpha = 0.15) 
    ax.set_title("Neuron_" + str(n))
    ax.set_xticklabels([])
#    ax.set_theta_zero_location('N')
#    ax.legend(lista)
plt.tight_layout()
#plt.gcf().text(0.5, 0.01, str(minutes) + ' min')
plt.suptitle("Control vs Stimulation ", x = 0.5, y = 1)
plt.savefig(data_directory + '/plots' + '/tun_baseVSstim_tot'  + '.pdf', bbox_inches = 'tight')

#Individual
neuron=18
lista=["Control","Stimulation"]
plt.figure(figsize=(8,8))
plt.polar(tuning_curves_base[neuron], color ='black')
#ax.fill(tuning_curves_base[n],"black", alpha = 0.15) 
plt.polar(tuning_curves_stim[neuron], color ='lime')
#ax.fill(tuning_curves_stim[n],"lime", alpha = 0.15) 
plt.title("Neuron_" + str(neuron))
plt.legend(lista)
plt.savefig(data_directory + '/plots' + '/tun_baseVSstim_' + str(n) + '.pdf', bbox_inches = 'tight') 

#All, individually 
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



"""
Raster plots
"""

#A. Restricted to preferred firing direction, per neuron. 
##Get the time intervals of the stimulation where the neuron was firing in its preferred direction
angle = position['ry'].realign(ttl_opto_start)
angle= pd.DataFrame(data=angle.values, index=angle.index.values, columns=['angle'])
limit_inf=(2*np.pi/8)*1
limit_sup=(2*np.pi/8)*7.5
angle['label']=(angle['angle'] >= limit_inf) & (angle['angle']<= limit_sup)
angle=angle[angle['label']]
#Use these intervals to restrict the time of the spikes
neuron = 8
spikes_list = []
for i in range(len(angle.index)):
    interval = nts.IntervalSet(start=angle.index[i] - 1500000 , end=angle.index[i]+ 2000000)
    t = spikes[neuron].restrict(interval).index.values - angle.index[i]*1
    spikes_list.append(t)
lineSize=0.5
left, bottom, width, height = (0, 0, 5000, len(angle.index))
rect = plt.Rectangle((left, bottom), width, height, facecolor="limegreen", alpha=0.1)
fig, ax = plt.subplots()
ax.add_patch(rect)
ax.eventplot(spikes_list, linelengths = 30, color='black')
ax.set_ylabel('Trials')
ax.set_xlabel('Time (us)')
ax.legend(["Period of stimulation"])
ax.set_title("Raster plot for all the trials")


#Find the beginning and end of the stimulation
t, _ = scipy.signal.find_peaks(np.diff(ttl_opto_start.as_units('s').index), height = 1)
stim_ep = np.sort(np.hstack(([ttl_opto_start.index[0]], ttl_opto_start.index[t], ttl_opto_start.index[t+1], ttl_opto_end.index[-1])))
stim_ep = stim_ep.reshape(len(stim_ep)//2, 2)
stim_ep = nts.IntervalSet(start = stim_ep[:,0], end = stim_ep[:,1])
plt.figure()
plot(ttl_opto_start.index)
[axhline(stim_ep.loc[i,'start']) for i in stim_ep.index]
[axhline(stim_ep.loc[i,'end']) for i in stim_ep.index]
plt.plot(ttl_opto_start.index)
plt.ylabel("time")
plt.title("Stimulation epochs at different intensities")
plt.show()

# B Take the first n periods of stimulation and plot them
n=2
stim_ep.loc[0:n]
neuron = 7
spikes_list = []
for i in range(n):
    print(i)
    interval = nts.IntervalSet(start=stim_ep['start'][i] - 2000000 , end=stim_ep['end'][i]+2000000)
    print(interval)
    t = spikes[neuron].restrict(interval).index.values - stim_ep['start'][i]
    spikes_list.append(t)
fig, (ax1,ax2) = plt.subplots(2, 1, sharex=True)
ax1.eventplot(spikes[neuron].restrict(interval).as_units('s').index)
ax2.hist(spikes[neuron].restrict(interval).as_units('s').index, bins = 10)
plt.show()

# C For all the stimulation epoch
neuron = 7
spikes_list = []
for i in range(len(opto_ep)):
    print(i)
    interval = nts.IntervalSet(start=opto_ep['start'][i] - 20000 , end=opto_ep['start'][i]+20000)
    print(interval)
    t = spikes[neuron].restrict(interval).index.values - opto_ep['start'][i]
    spikes_list.append(t)
np.save("{}{}".format(data_directory, "/Analysis/spikes_all"), spikes_list)

fig, (ax1,ax2) = plt.subplots(2, 1, sharex=True)
lineSize=0.5
left, bottom, width, height = (0, 0, 5000, len(spikes_list_h))
rect = plt.Rectangle((left, bottom), width, height, facecolor="limegreen", alpha=0.1)
ax1.add_patch(rect)
ax1.eventplot(spikes_list_h, linelengths = 30, color='black')

plt.plot()
plt.eventplot(spikes_list_h)
plt.show()
ax1.set_ylabel('Trials')
ax1.set_xlabel('Time (us)')
ax1.legend(["Period of stimulation"])
ax1.set_title("Raster plot for all the trials")
ax2.hist(spikes_list)
plt.savefig('{}{}{}{}'.format(data_directory, '/plots', '/spikes_all', '.pdf'))

# D Just one intensity
neuron = 7
time_stim = 5000 #us
timepre = 20000 
timepost = 20000
##Create an array of spikes restricted to some interval
interval=stim_ep.loc[[2]]
epoch = nts.Tsd(t=opto_ep.start.values)
epoch = epoch.restrict(interval).index.values
spikes_list_h = []
for i in range(len(epoch)):
    interval = nts.IntervalSet(start = epoch[i] - timepre , end = epoch[i]+timepost)
    t = spikes[neuron].restrict(interval).index.values - epoch[i]
    spikes_list_h.append(t)
np.save("{}{}".format(data_directory, "/Analysis/spikes_high"), spikes_list_h)
##
array = np.concatenate(spikes_list_h).ravel()
#bins = int(np.cbrt(len(spikes_list_h))) #needs to be even
bins=20
pre=int((timepre/1000)/((timepre/1000)*2)*bins) 
stim= int((time_stim/1000)/((timepre/1000)*2)*bins) 
post=pre-stim
bstim = np.concatenate ([np.zeros(pre), np.ones(stim), np.zeros(post)])
bspks, _ = np.histogram(array, bins=bins)
#Bar plot comparing 5ms previous with 5ms after
plt.figure()
x = bspks[pre-stim:pre]
y = bspks[pre:pre+stim]
data=pd.DataFrame(data=np.vstack([x,y]),columns=["pre (-5 ms)","stim (5ms)"])
data.index.name="L"
sns.barplot(data=data)
plt.ylabel("spikes")
plt.show()
##Raster
cestim="lightcyan"
cctrl="tan"
fig, (ax1,ax2) = plt.subplots(2, 1, sharex=True)
lineSize=0.5
left, bottom, width, height = (0, 0, time_stim, len(spikes_list_h))
rect = plt.Rectangle((left, bottom), width, height, facecolor=cestim)
#ax1.add_patch(rect)
ax1.eventplot(spikes_list_h, linelengths = 30, color='black')
ax1.set_ylabel('Trials')
ax1.legend(["Light"])
ax1.set_title("Raster plot for all the trials")
ax2.hist(array, bins=bins, color=cctrl)
ax2.set_xlabel('Time (us)')
ax2.set_ylabel('spikes')
plt.show()
plt.savefig('{}{}{}{}'.format(data_directory, '/plots', '/spikes_eph', '.pdf'))
#ax2.hist(spikes[neuron].restrict(interval).as_units('s').index, bins = 10)

"""
Firing rate
"""
MFirRate1 = computeMeanFiringRate(spikes, [wake_base],["base"])
MFirRate2 = computeMeanFiringRate(spikes, [wake_firststim],["stim"])
MFirRate = pd.concat ([MFirRate1, MFirRate2], axis=1)
plt.figure()
MFirRate.plot.bar()
plt.ylabel("spks/s")
plt.xlabel("neuron")
plt.savefig(data_directory + '/plots' + '/firing_rate' + '.pdf', bbox_inches = 'tight')


epoch_int = nts.IntervalSet(start=epoch[0], end=epoch[-1])
MFirRate1 = computeMeanFiringRate(spikes, [wake_base],["base"])
MFirRate2 = computeMeanFiringRate(spikes, [epoch_int],["stim"])
MFirRate = pd.concat ([MFirRate1, MFirRate2], axis=1)
MFirRate.plot.bar()








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

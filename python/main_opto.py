#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 19 11:40:41 2019

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
from scipy.stats import pearsonr
from vitemod import *

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

"""
Comparison of different intensities based on mean firing rate
"""
plotk = "_meanFR_"
#A by mean firing rate
#baseline = nts.IntervalSet(start=wake_ep.start, end=opto_ep.start[0]-1000000)
baseline=nts.IntervalSet(start = opto_ep.start[0]-2*60*1000*1000 - 1000000, 
                         end=opto_ep.start[0]-1000000)
FR=computeMeanFiringRate(spikes, [baseline, 
                                  stim_ep.loc[[0]], stim_ep.loc[[1]], stim_ep.loc[[2]]], 
                                  ["baseline", "low","med","high"])
plt.figure()
for i in range(len(FR.index)):
    plt.plot([FR["baseline"].values[i], FR["low"].values[i], FR["med"].values[i], FR["high"].values[i]], 
             'o-', c='black', linewidth = 0.1, alpha=0.7)
plt.title(session)
plt.xticks([0,1,2,3], ["Baseline","Low","Medium","High"])
plt.xlabel("Intensities")
plt.ylabel("firing rate")
plt.savefig('{}{}{}{}'.format(data_directory, '/plots/', session + plotk + "firing", '.pdf'))

#as a % of change respect to the baseline
FRp=(FR.loc[:,"low":"high"]*100).div(FR["baseline"], axis=0)-100
FRp.sort_values(by="high", inplace=True, ascending = False)
index = FRp.index.values
FRp.index = [*spikes.keys()]
colors = ["lightblue", "deepskyblue", "royalblue"]
FRp.plot(marker='o', color =colors, title=session)
plt.xticks([*spikes.keys()], index)
plt.ylabel("Relative Change")
plt.xlabel("Neurons")
plt.savefig('{}{}{}{}'.format(data_directory, '/plots/', session + plotk + "relativechange", '.pdf'))
FRp.index = index

plt.figure()
for i in range(len(FR.index)):
    b=FR["baseline"].values[i]
    plt.plot([FRp["low"].values[i], FRp["med"].values[i], FRp["high"].values[i]], 
             'o-', c='black', linewidth = 0.1, alpha=0.7)
plt.title(session)
plt.xticks([0,1,2], ["Low","Medium","High"])
plt.xlabel("Intensities")
plt.ylabel("Relative change")
plt.savefig('{}{}{}{}'.format(data_directory, '/plots/', session + plotk + "relativechangeb", '.pdf'))

#select neurons for a given cutoff
cutoff = -10
condition = ((FRp["low"]<cutoff)&(FRp["med"]<cutoff)&(FRp["high"]<cutoff)).values
neurons = FRp[condition].index.values
nspikes = {key:val for key, val in spikes.items() if key in neurons}
#try it for high intensity
condition = FRp["high"]<cutoff
neurons = FRp[condition].index.values
nspikesh = {key:val for key, val in spikes.items() if key in neurons}

"""
Rasters
"""

interval = stim_ep.loc[[2]]
span=8*60*1000*1000
lista = raster.gendata(nspikesh, span, [stim_ep['start'][2]])
stimduration = interval.tot_length('s')
binsize=1
data, edges = raster.histplot(lista, "test", "s", binsize, stimduration, session)
plt.savefig('{}{}{}{}'.format(data_directory, '/plots/', "test2", '.pdf'))

"""
matrix
"""
#A by firing rate
pre = 2*60*1000*1000
post = 4*60*1000*1000
step=1*1000*1000

for lap, l in zip ([*stim_ep['start']], label):
    spikes_df = matrix.gendata(spikes, pre, post, [lap], "s", step)
    #Order based on higher firing rate during the stimulation epoch
    dic = {key:val for key, val in enumerate(spikes_df.loc[0:120000000].sum())}
    order = pd.DataFrame(data=[dic.keys(),dic.values()]).transpose().sort_values(
            by=1, ascending=False)[0]
    spikes_df = spikes_df[order]
    spikes_df.rename(index = lambda s: int(s/1000000), inplace = True)
    #plot
    lista=[]
    for i in arange(-120,240,1):
        if i in rango:
            lista.append(i)
        else:
            lista.append("")
    fig = plt.figure(figsize = (20, 15))
    ax=fig.add_subplot(111,label="1")
    ax = sns.heatmap(spikes_df.transpose(), cmap= "coolwarm", xticklabels=lista)
    plt.title(session + " {}".format(l))
    plt.show()
    plt.tight_layout()
    plt.savefig('{}{}{}{}'.format(data_directory, '/plots/', session+"_matrix_"+l, '.pdf'))

#B as a % of change respect to the baseline
pre = 2*60*1000*1000
post = 4*60*1000*1000
step=10*1000*1000
label = ["Low", "Med", "High"]

for lap, l in zip ([*stim_ep['start']], label):
    #determine the mean firing rate previos to the stimulation epoch per neuron
    intervalb = nts.IntervalSet(start = lap -pre -1000*1000, end = lap -1000*1000)
    FRbase=computeMeanFiringRate(spikes, [intervalb], ['base'])
    #generate data frame
    spikes_dfb = matrix.gendatab(spikes, pre, post, [lap], step, FRbase)
    #Order the neurons based on higher firing rate during the stimulation epoch
    dic = {key:val for key, val in enumerate(spikes_dfb.loc[0:120000000].sum())}
    order = pd.DataFrame(data=[dic.keys(),dic.values()]).transpose().sort_values(
            by=1, ascending=False)[0]
    spikes_dfb = spikes_dfb[order]
    spikes_dfb.rename(index = lambda s: int(s/1000000), inplace = True)
    #plot
    rango = [*arange(-120,240,40)]
    lista=[]
    for i in arange(-120,240, 10):
        if i in rango:
            lista.append(i)
        else:
            lista.append("")
    fig = plt.figure(figsize = (20, 15))
    ax=fig.add_subplot(111,label="1")
    ax = sns.heatmap(spikes_dfb.transpose(), cmap= "coolwarm", xticklabels=lista, center = 0, cbar_kws={'label': 'firing rate as % change'})
    ax.set_xlabel("time (s)")
    ax.set_ylabel("neuron")
    ax2=ax.twinx()
    ax2.axvline(ax2.get_xlim()[1]/3, color='white', linewidth=1, zorder=-1)
    ax2.axvline(2*ax2.get_xlim()[1]/3, color='white', linewidth=1, zorder=-1)
    plt.title(session+" {}".format(l))
    plt.show()
    plt.tight_layout()
    plt.savefig('{}{}{}{}'.format(data_directory, '/plots/', session+"_matrixb_"+l , '.pdf'))


"""
S h a n k
"""

"""
Clusters
"""

lut = dict(zip([1,2,3,4], ['lightblue', 'steelblue', 'royalblue', 'midnightblue']))
shanks = pd.Series(shank)
rcolors = shanks.map(lut)
g = sns.clustermap(spikes_dfb.transpose(), 
                    col_cluster = True, 
                    xticklabels=xticks, 
                    cmap= "coolwarm", 
                    center = 0, 
                    dendrogram_ratio=(.1, .2),
                    row_colors=rcolors,
                    cbar_kws={'label': 'firing rate as % change'},
                    )
sns.despine(right = True)
g.fig.suptitle(session + " Clustermap " + l)
ax = g.ax_heatmap
ax.set_xlabel("time (s)")
ax.set_ylabel("neurons")
ax.axvline(ax2.get_xlim()[1]/3, color='white', linewidth=1, zorder=-1)
ax.axvline(2*ax2.get_xlim()[1]/3, color='white', linewidth=1, zorder=-1)
ax2=ax.twinx()
ax2.axvline(ax2.get_xlim()[1]/3, color='white', linewidth=1)
ax2.axvline(2*ax2.get_xlim()[1]/3, color='white', linewidth=1)

g.savefig('{}{}{}{}'.format(data_directory, '/plots/', session+"_clustermap_"+l , '.pdf'))

"""
"""
lut = dict(zip([1,2,3,4], ['lightblue', 'steelblue', 'royalblue', 'midnightblue']))
shanks = pd.Series(shank)
rcolors = shanks.map(lut)

df_estim = spikes_dfb.sort_index(axis=1).loc[0:120]
df_estim.columns = [shanks.values, shanks.index.values]
df_estim.columns.names = ["shank", "neuron"]
df_estim.index.name = "time"

cmap = sns.cubehelix_palette(dark=.1, light=.9, as_cmap=True)
sns.set(style="whitegrid")
sns.set_context('paper')
plt.figure()
sns.boxenplot(data = df_estim)
sns.stripplot(data = df_estim[1], color = 'k')
#sns.barplot(x = "neuron", y = "time", data=df_estim, hue = 'shank')
plt.show()

f, ax = plt.subplots()
sns.stripplot(data = df_estim[1], color = 'k', ax= axes[0,0])
sns.boxenplot(data = df_estim, ax= axes[0,0])

fig1 = plt.figure()
ax1 = fig1.add_subplot(1, 1, 1)
sns.stripplot(data = df_estim[1], color = 'k', ax= ax1)
sns.boxenplot(data = df_estim, ax= ax1)

"""
Compare cells
"""
plt.figure()
plt.plot(edges[0:-1],data)
plt.show()
for i in spikes:
    rasters[i] = {}
    for j,t in enumerate(ttl_opto.index.values):
        tmp = spikes[i].loc[t-2*1e6:t+3*1e6]
        tmp.index -= t
        if len(tmp):
            rasters[i][j] = tmp.fillna(j)
    rasters[i] = pd.DataFrame.from_dict(rasters[i])


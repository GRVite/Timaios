#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 11 12:33:03 2020

@author: vite
"""

import numpy as np
import pandas as pd
import neuroseries as nts
from pylab import *
from wrappers import *
from functions import *
import sys

data_directory 	= '/Volumes/LaCie/Timaios/Kilosorted/A3304/A3304-191130/A3304-191130'
data_directory_load = data_directory + '/my_data'
dir2save_plots = data_directory_load + '/plots'
if os.path.exists(dir2save_plots)==False:
    os.mkdir(dir2save_plots)

# load data
spikes = pickle.load(open(data_directory_load + '/spikes.pickle', 'rb'))
shank = pickle.load(open(data_directory_load  + '/shank.pickle', 'rb'))
episodes = pickle.load(open(data_directory_load + '/episodes.pickle', 'rb'))
position = pickle.load(open(data_directory_load  + '/position.pickle', 'rb'))
wake_ep = pickle.load(open(data_directory_load  + '/wake_ep.pickle', 'rb'))
sleep_ep = pickle.load(open(data_directory_load  + '/wake_ep.pickle', 'rb'))
      
n_channels, fs, shank_to_channel 	= loadXML(data_directory)
acceleration	= loadAuxiliary(data_directory)  
newsleep_ep 	= refineSleepFromAccel(acceleration, sleep_ep)
   
##################################################################################################
# DOWNSAMPLING
##################################################################################################
path = os.path.join(data_directory, 'dat.eeg')
if not os.path.exists(path):
    downsampleDatFile(path)
   
   
##################################################################################################
# LOADING LFP
##################################################################################################
lfp 		= loadLFP(path, n_channels, 12, 1250, 'int16')
lfp 		= downsample(lfp, 1, 5)
   
##################################################################################################
# DETECTION THETA
##################################################################################################

lfp_filt_theta	= nts.Tsd(lfp.index.values, butter_bandpass_filter(lfp, 5, 15, 1250/5, 2))
power_theta		= nts.Tsd(lfp_filt_theta.index.values, np.abs(lfp_filt_theta.values))
power_theta		= power_theta.rolling(window=1000,win_type='gaussian',center=True,min_periods=1).mean(std=40)

lfp_filt_delta	= nts.Tsd(lfp.index.values, butter_bandpass_filter(lfp, 1, 4, 1250/5, 2))
power_delta		= nts.Tsd(lfp_filt_delta.index.values, np.abs(lfp_filt_delta.values))
power_delta		= power_delta.rolling(window=1000,win_type='gaussian',center=True,min_periods=1).mean(std=40)

ratio 			= nts.Tsd(t = power_theta.index.values, d = np.log(power_theta.values/power_delta.values))
ratio2			= ratio.rolling(window=1000,win_type='gaussian',center=True,min_periods=1).mean(std=200)
ratio2 			= nts.Tsd(t = ratio2.index.values, d = ratio2.values)

# ratio2.as_series().to_hdf('../figures/figures_poster_2019/ratio2.h5', 'w')


# 	index 			= (ratio2.as_series() > 0).values*1.0
index 			= (ratio2 > 0).values*1.0
start_cand 		= np.where((index[1:] - index[0:-1]) == 1)[0]+1
end_cand 		= np.where((index[1:] - index[0:-1]) == -1)[0]
if end_cand[0] < start_cand[0]:	end_cand = end_cand[1:]
if end_cand[-1] < start_cand[-1]: start_cand = start_cand[0:-1]
tmp 			= np.where(end_cand != start_cand)
start_cand 		= ratio2.index.values[start_cand[tmp]]
end_cand	 	= ratio2.index.values[end_cand[tmp]]
good_ep			= nts.IntervalSet(start_cand, end_cand)
good_ep			= newsleep_ep.intersect(good_ep)
good_ep			= good_ep.drop_short_intervals(10, time_units = 's')
good_ep			= good_ep.reset_index(drop=True)
good_ep			= good_ep.merge_close_intervals(5, time_units = 's')

theta_rem_ep	= good_ep
sws_ep 	= newsleep_ep.set_diff(theta_rem_ep)
sws_ep = sws_ep.merge_close_intervals(0).drop_short_intervals(0)


plot(ratio.restrict(newsleep_ep))
plot(ratio2.restrict(newsleep_ep))
[plot(theta_rem_ep.loc[i], np.zeros(2)) for i in theta_rem_ep.index]
[plot(sws_ep.loc[i], np.ones(2)) for i in sws_ep.index]
show()
# sys.exit()

writeNeuroscopeEvents(os.path.join(data_directory,'rem.evt'), theta_rem_ep, "Theta")
writeNeuroscopeEvents(os.path.join(data_directory,'sws.evt'), sws_ep, "SWS")
sys.exit()

# phase 			= getPhase(lfp_hpc, 6, 14, 16, fs/5.)	
# ep 				= { 'wake'	: theta_wake_ep,
# 					'rem'	: theta_rem_ep}
# theta_mod 		= {}



# for e in ep.keys():		
# 	spikes_phase	= {n:phase.realign(spikes[n], align = 'closest') for n in spikes.keys()}

# 	# theta_mod[e] 	= np.ones((n_neuron,3))*np.nan
# 	theta_mod[e] 	= {}
# 	for n in range(len(spikes_phase.keys())):			
# 		neuron = list(spikes_phase.keys())[n]
# 		ph = spikes_phase[neuron].restrict(ep[e])
# 		mu, kappa, pval = getCircularMean(ph.values)
# 		theta_mod[e][session.split("/")[1]+"_"+str(neuron)] = np.array([mu, pval, kappa])
# 		spikes_theta_phase[e][session.split("/")[1]+"_"+str(neuron)] = ph.values


# stop = time.time()
# print(stop - start, ' s')		
# datatosave[session] = theta_mod


"""
Plots
"""

#Acceleration during wake
figure()

plt.title('Acceleration')
plot(acceleration[0].restrict(wake_ep))
plt.tight_layout
show()
plt.savefig(dir2save_plots + '/acceleration_wake.pdf')

# SWS
figure()

ax = subplot(311)
plt.title('LFP trace')
[plot(lfp.restrict(sws_ep.loc[[i]]), color = 'skyblue') for i in sws_ep.index]
plot(lfp_filt_delta.restrict(sws_ep), color = 'tan')

subplot(312, sharex = ax)
plt.title('Theta/Delta ratio')
[plot(ratio.restrict(sws_ep.loc[[i]]), color = 'skyblue') for i in sws_ep.index]
plot(ratio2.restrict(sws_ep), color = 'tan')
axhline(0)


subplot(313, sharex = ax)
plt.title('Acceleration')
plot(acceleration[0].restrict(sws_ep), color = 'steelblue')
show()

plt.savefig(dir2save_plots + '/sws.pdf')

#REM
figure()

ax = subplot(311)
plt.title('LFP trace')
[plot(lfp.restrict(theta_rem_ep.loc[[i]]), color = 'skyblue') for i in sws_ep.index]
plot(lfp_filt_theta.restrict(theta_rem_ep), color = 'tan')

subplot(312, sharex = ax)
plt.title('Theta/Delta ratio')
[plot(ratio.restrict(theta_rem_ep.loc[[i]]), color = 'skyblue') for i in sws_ep.index]
plot(ratio2.restrict(theta_rem_ep), color = 'tan')
axhline(0)

subplot(313, sharex = ax)
plt.title('Acceleration')
plot(acceleration[0].restrict(theta_rem_ep), color = 'skyblue')
plt.tight_layout
show()

plt.savefig(dir2save_plots + '/rem.pdf')



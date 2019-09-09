#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 19 15:53:19 2019
@author: grvite

You are expected to run Process_LFPfromDat.m on your .dat file before using this script
"""

from lfp import *
import numpy as np
import scipy.signal as ss
import matplotlib.pyplot as plt
import sys
import neuroseries as nts
    
#Load lfp's for all channels
lfp_whole = []
x= [loadLFP('/home/grvite/Data/A3301/A3301-190618/A3301-190618/A3301-190618.lfp',channel = i) for i in range(32)]

for i in range(32):
    x = loadLFP('/home/grvite/Data/A3301/A3301-190618/A3301-190618/A3301-190618.lfp',channel = i) 
    plt.plot(x)
    plt.show()

#Load lfp from desired channel
#x = loadLFP('/home/grvite/Data/A3301/A3301-190621/A3301-190621/A3301-190621.lfp',channel = 1) 
x = loadLFP('/home/grvite/Data/A3301/A3301-190618/A3301-190618/A3301-190618.lfp',channel = 6) 
plt.figure(1)
plt.plot(x)
plt.show()

#downsample data. E.g. (x,1,5) means that you will unsample your data by a factor of 5
factor = 5
sample = downsample(x, 1, factor)
plt.figure(2)
plt.plot(sample)
plt.show()

"""Plot your signal vs a smoothed version of it"""
sf = butter_bandpass_filter(sample.values.astype('float32'), 1, 10, 1250/factor, order=2)
sf = nts.Tsd(t = sample.index.values, d = sf)

#Whole
plt.figure(3)
plt.plot(sample.as_units('s'))
plt.plot(sf.as_units('s'),'r')
plt.show()
#Sample
plt.figure(4)
plt.plot(sample.as_units('s').loc[4277:4287])
plt.plot(sf.as_units('s').loc[4277:4287],'r')
plt.show()



"""Plot your signal vs a sine wave"""

## Whole
# Get x values of the sine wave.
time = np.arange(0, 6415.196, 0.004);
# Amplitude of the sine wave is sine of a variable like time.
amplitude = np.sin(time*40)
# Plot a sine wave using time and amplitude obtained for the sine wave.
plt.figure(5)
plt.plot(time, amplitude*1000)
plt.plot(sf.as_units('s').loc[0:6415.196],'orange')
plt.show()


## Sample
# Get x values of the sine wave.
time = np.arange(4277, 4287, 0.004);
# Amplitude of the sine wave is sine of a variable like time.
amplitude = np.sin(time*40)
# Plot a sine wave using time and amplitude obtained for the sine wave.
plt.figure(6)
plt.plot(time, amplitude*1000)
plt.plot(sf.as_units('s').loc[4277:4287],'orange')
plt.show()




"""Plot an spectrogram"""

#Uno

f, t, Sxx = ss.spectrogram(sf, 1250/5)
plt.figure(7)
plt.pcolormesh(t, f, Sxx, cmap ='winter_r')
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')
plt.colorbar()
plt.show()



#It works!
NFFT = 1024 
plt.figure(8)
tmp = plt.specgram(sf, Fs=1250/5, noverlap=250)
plt.ylabel('Frequency [Hz]')
plt.show()
plt.colorbar()


plt.figure(9)
tmp = plt.specgram(sf, Fs=1250/5)
plt.ylabel('Frequency [Hz]')
plt.show()
plt.colorbar()
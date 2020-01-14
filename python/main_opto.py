#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 19 11:40:41 2019

@author: grvite
"""




ttl_track, ttl_opto = loadTTLPulse2(os.path.join(data_directory, session+'_0_analogin.dat'), 2)

ttl_opto = nts.Ts(ttl_opto.index.values, time_units = 's')

figure()
plt.plot(ttl_opto.index.values, 'o')
plt.show()

rasters = {}

for i in spikes:
    rasters[i] = {}
    for j,t in enumerate(ttl_opto.index.values):
        tmp = spikes[i].loc[t-2*1e6:t+3*1e6]
        tmp.index -= t
        if len(tmp):
            rasters[i][j] = tmp.fillna(j)
    rasters[i] = pd.DataFrame.from_dict(rasters[i])

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

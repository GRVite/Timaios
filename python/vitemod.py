import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pylab import *
import neuroseries as nts
import sys
from scipy.signal import find_peaks




class manage:
    def __init__(self):
        return

    def optoeps(ttl_start,ttl_end, height = 1):
        #Find the beginning and end of the stimulation
        t, _ = find_peaks(np.diff(ttl_start.as_units('s').index), height)
        stim_ep = np.sort(np.hstack(([ttl_start.index[0]], ttl_start.index[t], ttl_start.index[t+1], ttl_end.index[-1])))
        stim_ep = stim_ep.reshape(len(stim_ep)//2, 2)
        stim_ep = nts.IntervalSet(start = stim_ep[:,0], end = stim_ep[:,1])
        plt.figure()
        plot(ttl_start.index)
        [axhline(stim_ep.loc[i,'start']) for i in stim_ep.index]
        [axhline(stim_ep.loc[i,'end']) for i in stim_ep.index]
        plt.plot(ttl_start.index)
        plt.ylabel("time")
        plt.title("Stimulation epochs at different intensities")
        plt.show()
        return stim_ep

class raster:
    def __init__(self):
        return
    
    def gendata(spk, span, lapsos):
        timep=int(span/2)
        spikes_list=[]
        for neuron in spk.keys():
            for i in lapsos:
                interval = nts.IntervalSet(start = i - timep , end = i+timep)
                t = spk[neuron].restrict(interval).index.values - i
                spikes_list.append(t)
        return  spikes_list
    
    def gendatans(spikes, epoch, span):
        timep=span/2
        for i in range(epoch[0],epoch[-1], span):
            interval = nts.IntervalSet(start = i - timep , end = i+timep)
            t = spikes[neuron].restrict(interval).index.values - i
            spikes_list.append(t)
        return  spikes_list

    
    def histplot(lista, name2save, units, binsize, width, session, ylabel = "Firing Rate", linesize=0.5,cestim="lightcyan", cctrl="tan"):
        if units == 'ms':
            scale = 1000
        elif units == 's':
            scale = 1000000
        else:
            return print("wrong units input")
        left, bottom, height = (0, 0, len(lista))
        rect = plt.Rectangle((left, bottom), width, height, facecolor=cestim, alpha=0.5)
        lista = [i/scale for i in lista]
        fig, (ax1,ax2) = plt.subplots(2, 1, sharex=False)
        ax1.add_patch(rect)
        ax1.eventplot(lista, linelengths = linesize, color='black')
        ax1.set_ylabel('Trials')
        ax1.legend(["Light"])
        ax1.set_title(session)
        ax1.set_frame_on(False)
        array = np.concatenate(lista).ravel()
        nbins=int(array[-1]*2/binsize)
        data, edges = np.histogram(array,nbins)
        ax2.hist(array, bins=nbins, color=cctrl)
        ax2.plot([0,100],[width, width], linewidth=7, color=cestim)
        ax2.set_xlabel("{}{}".format('Time (',units+')'))
        ax2.set_ylabel(ylabel)
        ax2.set_frame_on(False)
        plt.show()
        return data,edges
#        

class matrix:
    def __init__(self):
        return
    
    def gendata(spk, pre, post, lapsos, units, step):
        bins=arange(pre*-1,post,step)
        spikes_df=pd.DataFrame(columns=spk.keys(), index=bins)
        for neuron in spk.keys():
            for i in lapsos:
                interval = nts.IntervalSet(start = i - pre , end = i + post)
                t = spk[neuron].restrict(interval).index.values - i
                bins=arange(pre*-1,post+step,step)
                t, edges = np.histogram(t,bins)
                spikes_df[neuron]=t
        return  spikes_df
    
    def gendatab(spk, pre, post, lapsos, step, FRbase):
        bins=arange(pre*-1,post,step)
        spikes_dfb=pd.DataFrame(columns=spk.keys(), index=bins)
        for neuron in spk.keys():
            for i in lapsos:
                print(neuron,i)
                interval = nts.IntervalSet(start = i - pre , end = i + post)
                t = spk[neuron].restrict(interval).index.values - i
                bins=arange(pre*-1,post+step,step)
                t, edges = np.histogram(t,bins)
                t = t / 10
                per = t*100/FRbase.iloc[neuron][0]-100
                spikes_dfb[neuron]= per
        return  spikes_dfb
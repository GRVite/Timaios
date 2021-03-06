import numpy as np
from numba import jit
import pandas as pd
import neuroseries as nts
import sys
from scipy.signal import find_peaks
'''
Utilities functions
Feel free to add your own
'''

###
# Extra
###

def path_spk_plt(ep,spikes,position):
    fig = figure(figsize = (15,16))
    fig.suptitle('Spikes + Path Plot',size=30)
    for i in spikes:
        ax=subplot(4,5,i+1) #if you have more than 20cells change the numbers in bracket to reflect that
        scatter(position['x'].realign(spikes[i].restrict(ep)),position['z'].realign(spikes[i].restrict(ep)),s=5,c='magenta',label=str(i))
        legend()
        plot(position['x'].restrict(ep),position['z'].restrict(ep),color='darkgrey', alpha=0.5)  
    return fig,ax

def smooth_corr(aucorr, nbins, binsize, meanfiring, window = 7, stdv = 5.0, plot = False):
    aucorr= aucorr-meanfiring
    dfa = aucorr [0:int(nbins/2)]
    dfa = pd.DataFrame(dfa).rolling(window = window, win_type='gaussian', center=True, min_periods = 1).mean(std = stdv)
    dfb = np.flipud(aucorr [int(nbins/2)+1::])
    dfb = pd.DataFrame(dfb).rolling(window = window, win_type='gaussian', center=True, min_periods = 1).mean(std = stdv)
    #array = np.append((dfa.values),0)
    arrayt = np.append(np.append((dfa.values),0), np.flipud(dfb.values))
    if plot == True: 
        #Make a Tsd
        times = np.arange(0, binsize*(nbins+1), binsize) - (nbins*binsize)/2
        ndf = nts.Tsd(t = times, d = arrayt/meanfiring)
        ndf.plot()
    return arrayt
#########################################################
# CORRELATION
#########################################################
@jit(nopython=True)
def crossCorr(t1, t2, binsize, nbins):
    ''' 
        Fast crossCorr 
    '''
    nt1 = len(t1)
    nt2 = len(t2)
    if np.floor(nbins/2)*2 == nbins:
        nbins = nbins+1

    m = -binsize*((nbins+1)/2)
    B = np.zeros(nbins)
    for j in range(nbins):
        B[j] = m+j*binsize

    w = ((nbins/2) * binsize)
    C = np.zeros(nbins)
    i2 = 1

    for i1 in range(nt1):
        lbound = t1[i1] - w
        while i2 < nt2 and t2[i2] < lbound:
            i2 = i2+1
        while i2 > 1 and t2[i2-1] > lbound:
            i2 = i2-1

        rbound = lbound
        l = i2
        for j in range(nbins):
            k = 0
            rbound = rbound+binsize
            while l < nt2 and t2[l] < rbound:
                l = l+1
                k = k+1

            C[j] += k

    # for j in range(nbins):
    # C[j] = C[j] / (nt1 * binsize)
    C = C/(nt1 * binsize/1000)

    return C

def crossCorr2(t1, t2, binsize, nbins):
    '''
        Slow crossCorr
    '''
    window = np.arange(-binsize*(nbins/2),binsize*(nbins/2)+2*binsize,binsize) - (binsize/2.)
    allcount = np.zeros(nbins+1)
    for e in t1:
        mwind = window + e
        # need to add a zero bin and an infinite bin in mwind
        mwind = np.array([-1.0] + list(mwind) + [np.max([t1.max(),t2.max()])+binsize])    
        index = np.digitize(t2, mwind)
        # index larger than 2 and lower than mwind.shape[0]-1
        # count each occurences 
        count = np.array([np.sum(index == i) for i in range(2,mwind.shape[0]-1)])
        allcount += np.array(count)
    allcount = allcount/(float(len(t1))*binsize / 1000)
    return allcount

def xcrossCorr_slow(t1, t2, binsize, nbins, nbiter, jitter, confInt):        
    times             = np.arange(0, binsize*(nbins+1), binsize) - (nbins*binsize)/2
    H0                 = crossCorr(t1, t2, binsize, nbins)    
    H1                 = np.zeros((nbiter,nbins+1))
    t2j                 = t2 + 2*jitter*(np.random.rand(nbiter, len(t2)) - 0.5)
    t2j             = np.sort(t2j, 1)
    for i in range(nbiter):            
        H1[i]         = crossCorr(t1, t2j[i], binsize, nbins)
    Hm                 = H1.mean(0)
    tmp             = np.sort(H1, 0)
    HeI             = tmp[int((1-confInt)/2*nbiter),:]
    HeS             = tmp[int((confInt + (1-confInt)/2)*nbiter)]
    Hstd             = np.std(tmp, 0)

    return (H0, Hm, HeI, HeS, Hstd, times)

def xcrossCorr_fast(t1, t2, binsize, nbins, nbiter, jitter, confInt):        
    times             = np.arange(0, binsize*(nbins*2+1), binsize) - (nbins*2*binsize)/2
    # need to do a cross-corr of double size to convolve after and avoid boundary effect
    H0                 = crossCorr(t1, t2, binsize, nbins*2)    
    window_size     = 2*jitter//binsize
    window             = np.ones(window_size)*(1/window_size)
    Hm                 = np.convolve(H0, window, 'same')
    Hstd            = np.sqrt(np.var(Hm))    
    HeI             = np.NaN
    HeS             = np.NaN    
    return (H0, Hm, HeI, HeS, Hstd, times)    

def compute_AutoCorrs(spks, ep, binsize = 5, nbins = 200):
    # First let's prepare a pandas dataframe to receive the data
    times = np.arange(0, binsize*(nbins+1), binsize) - (nbins*binsize)/2    
    autocorrs = pd.DataFrame(index = times, columns = np.arange(len(spks)))
    firing_rates = pd.Series(index = np.arange(len(spks)))

    # Now we can iterate over the dictionnary of spikes
    for i in spks:
        # First we extract the time of spikes in ms during wake
        spk_time = spks[i].restrict(ep).as_units('ms').index.values
        # Calling the crossCorr function
        autocorrs[i] = crossCorr(spk_time, spk_time, binsize, nbins)
        # Computing the mean firing rate
        firing_rates[i] = len(spk_time)/ep.tot_length('s')

    # We can divide the autocorrs by the firing_rates
    autocorrs = autocorrs / firing_rates

    # And don't forget to replace the 0 ms for 0
    autocorrs.loc[0] = 0.0
    return autocorrs, firing_rates

def compute_AutoCorrs_G(spks, binsize = 5, nbins = 200):
    # First let's prepare a pandas dataframe to receive the data
    times = np.arange(0, binsize*(nbins+1), binsize) - (nbins*binsize)/2    
    autocorrs = pd.DataFrame(index = times, columns = np.arange(len(spks)))
    firing_rates = pd.Series(index = np.arange(len(spks)))

    # Now we can iterate over the dictionnary of spikes
    for i in spks:
        # First we extract the time of spikes in ms during wake
        spk_time = spks[i].as_units('ms').index.values
        # Calling the crossCorr function
        autocorrs[i] = crossCorr(spk_time, spk_time, binsize, nbins)

    # We can divide the autocorrs by the firing_rates
    autocorrs = autocorrs / firing_rates

    # And don't forget to replace the 0 ms for 0
    autocorrs.loc[0] = 0.0
    return autocorr

#########################################################
# VARIOUS
#########################################################
    

def loadTTLPulse2(file, n_channels = 2, track = 0, opto = 1, fs = 20000):
    """
        load ttl from analogin.dat
    """
    f = open(file, 'rb')
    startoffile = f.seek(0, 0)
    endoffile = f.seek(0, 2)
    bytes_size = 2        
    n_samples = int((endoffile-startoffile)/n_channels/bytes_size)
    f.close()
    with open(file, 'rb') as f:
        data = np.fromfile(f, np.uint16).reshape((n_samples, n_channels))
    
    ch_track = data[:,track].astype(np.int32)
    peaks,_ = find_peaks(np.diff(ch_track), height=30000)
    timestep = np.arange(0, len(data))/fs
    peaks+=1
    ttl_track = pd.Series(index = timestep[peaks], data = data[peaks,track])    

    ch_opto = data[:,opto].astype(np.int32)
    peaks,_ = find_peaks(np.diff(ch_opto), height=5000)
    trough,_ = find_peaks(np.diff(ch_opto)*-1, height=5000)
    peaks+=1
    trough+=1
    ttl_opto_start = pd.Series(index = timestep[peaks], data = data[peaks,opto])
    ttl_opto_end = pd.Series(index = timestep[trough], data = data[trough,opto])

    return ttl_track, ttl_opto_start, ttl_opto_end

def computeAngularTuningCurves(spikes, angle, ep, nb_bins = 180, frequency = 120.0):
    bins             = np.linspace(0, 2*np.pi, nb_bins)
    idx             = bins[0:-1]+np.diff(bins)/2
    tuning_curves     = pd.DataFrame(index = idx, columns = np.arange(len(spikes)))    
    angle             = angle.restrict(ep)
    # Smoothing the angle here
    # tmp             = pd.Series(index = angle.index.values, data = np.unwrap(angle.values))
    # tmp2             = tmp.rolling(window=50,win_type='gaussian',center=True,min_periods=1).mean(std=10.0)
    # angle            = nts.Tsd(tmp2%(2*np.pi))
    for k in spikes:
        spks             = spikes[k]
        # true_ep         = nts.IntervalSet(start = np.maximum(angle.index[0], spks.index[0]), end = np.minimum(angle.index[-1], spks.index[-1]))        
        spks             = spks.restrict(ep)    
        angle_spike     = angle.restrict(ep).realign(spks)
        spike_count, bin_edges = np.histogram(angle_spike, bins)
        occupancy, _     = np.histogram(angle, bins)
        spike_count     = spike_count/occupancy        
        tuning_curves[k] = spike_count*frequency    

    return tuning_curves

def findHDCells(tuning_curves):
    """
        Peak firing rate larger than 1
        and Rayleigh test p<0.001 & z > 100
    """
    cond1 = tuning_curves.max()>1.0
    
    from pycircstat.tests import rayleigh
    stat = pd.DataFrame(index = tuning_curves.columns, columns = ['pval', 'z'])
    for k in tuning_curves:
        stat.loc[k] = rayleigh(tuning_curves[k].index.values, tuning_curves[k].values)
    cond2 = np.logical_and(stat['pval']<0.001,stat['z']>20)
    tokeep = np.where(np.logical_and(cond1, cond2))[0]
    return tokeep, stat

def decodeHD(tuning_curves, spikes, ep, px, bin_size = 200):
    """
        See : Zhang, 1998, Interpreting Neuronal Population Activity by Reconstruction: Unified Framework With Application to Hippocampal Place Cells
        tuning_curves: pd.DataFrame with angular position as index and columns as neuron
        spikes : dictionnary of spike times
        ep : nts.IntervalSet, the epochs for decoding
        bin_size : in ms (default:200ms)
        px : Occupancy. If None, px is uniform
    """        
    if len(ep) == 1:
        bins = np.arange(ep.as_units('ms').start.iloc[0], ep.as_units('ms').end.iloc[-1], bin_size)
    else:
        print("TODO, more than one epoch")
        sys.exit()
    
    spike_counts = pd.DataFrame(index = bins[0:-1]+np.diff(bins)/2, columns = tuning_curves.columns)
    for k in spike_counts.columns:
        spks = spikes[k].restrict(ep).as_units('ms').index.values
        spike_counts[k], _ = np.histogram(spks, bins)

    tcurves_array = tuning_curves.values
    spike_counts_array = spike_counts.values
    proba_angle = np.zeros((spike_counts.shape[0], tuning_curves.shape[0]))

    part1 = np.exp(-(bin_size/1000)*tcurves_array.sum(1))    
    part2 = px
    
    for i in range(len(proba_angle)):
        part3 = np.prod(tcurves_array**spike_counts_array[i], 1)
        p = part1 * part2 * part3
        proba_angle[i] = p/p.sum() # Normalization process here

    proba_angle  = pd.DataFrame(index = spike_counts.index.values, columns = tuning_curves.index.values, data= proba_angle)    
    # proba_angle = proba_angle.astype('float')        
    decoded = nts.Tsd(t = proba_angle.index.values, d = proba_angle.idxmax(1).values, time_units = 'ms')
    return decoded, proba_angle

def firingMap(spikes, position, ep):
    position_tsd = position.restrict(ep)
    pos_a=position_tsd.index[0]
    pos_b=position_tsd.index[-1]
    my_neuron = spikes[7].restrict(ep)
    first_spike = my_neuron.index[0]
    last_spike = my_neuron.index[-1]
    #Determine bin size in us
    bin_size=1000000 # = 1s
    # Observe the -1 for the value at the end of an array
    duration = last_spike - first_spike
    # it's the time of the last spike
    # with a bin size of 1 second, the number of points is 
    nb_points = duration/bin_size  
    nb_points = int(nb_points)


    last_spike =position_tsd.index[0]
    
    position_tsd.as_units('ms').index[0]
    spikes.as_units('ms').index[0]
def computePlaceFields(spikes, position, ep, nb_bins, frequency = 120.0):
    place_fields = {}
    position_tsd = position.restrict(ep)
    xpos = position_tsd.iloc[:,0]
    ypos = position_tsd.iloc[:,1]
    xbins = np.linspace(xpos.min(), xpos.max()+1e-6, nb_bins+1)
    ybins = np.linspace(ypos.min(), ypos.max()+1e-6, nb_bins+1)    
    for n in spikes:
        position_spike = position_tsd.realign(spikes[n].restrict(ep))
        spike_count,_,_ = np.histogram2d(position_spike.iloc[:,1].values, position_spike.iloc[:,0].values, [ybins,xbins])
        occupancy, _, _ = np.histogram2d(ypos, xpos, [ybins,xbins])
        mean_spike_count = spike_count/(occupancy+1)
        place_field = mean_spike_count*frequency    
        place_fields[n] = pd.DataFrame(index = ybins[0:-1][::-1],columns = xbins[0:-1], data = place_field)
        
    extent = (xbins[0], xbins[-1], ybins[0], ybins[-1]) # USEFUL FOR MATPLOTLIB
    return place_fields, extent

def computeOccupancy(position_tsd, nb_bins = 100):
    xpos = position_tsd.iloc[:,0]
    ypos = position_tsd.iloc[:,1]    
    xbins = np.linspace(xpos.min(), xpos.max()+1e-6, nb_bins+1)
    ybins = np.linspace(ypos.min(), ypos.max()+1e-6, nb_bins+1)
    occupancy, _, _ = np.histogram2d(ypos, xpos, [ybins,xbins])
    return occupancy

def downsample(tsd, up, down):
    import scipy.signal
    import neuroseries as nts
    dtsd = scipy.signal.resample_poly(tsd.values, up, down)
    dt = tsd.as_units('s').index.values[np.arange(0, tsd.shape[0], down)]
    if len(tsd.shape) == 1:        
        return nts.Tsd(dt, dtsd, time_units = 's')
    elif len(tsd.shape) == 2:
        return nts.TsdFrame(dt, dtsd, time_units = 's', columns = list(tsd.columns))
    
def getPhase(lfp, fmin, fmax, nbins, fsamp, power = False):
    """ Continuous Wavelets Transform
        return phase of lfp in a Tsd array
    """
    import neuroseries as nts
    from Wavelets import MyMorlet as Morlet
    if isinstance(lfp, nts.time_series.TsdFrame):
        allphase         = nts.TsdFrame(lfp.index.values, np.zeros(lfp.shape))
        allpwr             = nts.TsdFrame(lfp.index.values, np.zeros(lfp.shape))
        for i in lfp.keys():
            allphase[i], allpwr[i] = getPhase(lfp[i], fmin, fmax, nbins, fsamp, power = True)
        if power:
            return allphase, allpwr
        else:
            return allphase            

    elif isinstance(lfp, nts.time_series.Tsd):
        cw                 = Morlet(lfp.values, fmin, fmax, nbins, fsamp)
        cwt             = cw.getdata()
        cwt             = np.flip(cwt, axis = 0)
        wave             = np.abs(cwt)**2.0
        phases             = np.arctan2(np.imag(cwt), np.real(cwt)).transpose()    
        cwt             = None
        index             = np.argmax(wave, 0)
        # memory problem here, need to loop
        phase             = np.zeros(len(index))    
        for i in range(len(index)) : phase[i] = phases[i,index[i]]
        phases             = None
        if power: 
            pwrs         = cw.getpower()        
            pwr         = np.zeros(len(index))        
            for i in range(len(index)):
                pwr[i] = pwrs[index[i],i]    
            return nts.Tsd(lfp.index.values, phase), nts.Tsd(lfp.index.values, pwr)
        else:
            return nts.Tsd(lfp.index.values, phase)

def refineSleepFromAccel(acceleration, sleep_ep):
    vl = acceleration[0].restrict(sleep_ep)
    vl = vl.as_series().diff().abs().dropna()    
    a, _ = find_peaks(vl, 0.025)
    peaks = nts.Tsd(vl.iloc[a])
    duration = np.diff(peaks.as_units('s').index.values)
    interval = nts.IntervalSet(start = peaks.index.values[0:-1], end = peaks.index.values[1:])
    newsleep_ep = interval.iloc[duration>15.0]
    newsleep_ep = newsleep_ep.reset_index(drop=True)
    plt.figure()
    plt.plot(vl)
    plt.plot(newsleep_ep, '.')
    newsleep_ep = newsleep_ep.merge_close_intervals(100000, time_units ='us')

    newsleep_ep    = sleep_ep.intersect(newsleep_ep)

    return newsleep_ep

def smoothAngularTuningCurves(tuning_curves, window = 20, deviation = 3.0):
    for i in tuning_curves.columns:
        tcurves = tuning_curves[i]
        padded     = pd.Series(index = np.hstack((tcurves.index.values-(2*np.pi),
                                                tcurves.index.values,
                                                tcurves.index.values+(2*np.pi))),
                            data = np.hstack((tcurves.values, tcurves.values, tcurves.values)))
        smoothed = padded.rolling(window=window,win_type='gaussian',center=True,min_periods=1).mean(std=deviation)        
        tuning_curves[i] = smoothed[tcurves.index]

    return tuning_curves

def computeMeanFiringRate(spikes, epochs, name):
    mean_frate = pd.DataFrame(index = spikes.keys(), columns = name)
    for n, ep in zip(name, epochs):
        for k in spikes:
            mean_frate.loc[k,n] = len(spikes[k].restrict(ep))/ep.tot_length('s')
    return mean_frate

from sklearn.manifold import Isomap, TSNE
from pylab import *
from mpl_toolkits.mplot3d import Axes3D


def makeRingManifold(spikes, ep, angle, position, bin_size = 200):
    """
    spikes : dict of hd spikes
    ep : epoch to restrict
    angle : tsd of angular direction
    bin_size : in ms
    """
    neurons = np.sort(list(spikes.keys()))
    bins = np.arange(ep.as_units('ms').start.iloc[0], ep.as_units('ms').end.iloc[-1]+bin_size, bin_size)
    spike_counts = pd.DataFrame(index = bins[0:-1]+np.diff(bins)/2, columns = neurons)
    for i in neurons:
        spks = spikes[i].as_units('ms').index.values
        spike_counts[i], _ = np.histogram(spks, bins)

    rates = np.sqrt(spike_counts/(bin_size))
#    rates = spike_counts/(bin_size)
    
    # BIN ANGLE
    angle = angle.restrict(ep)
    newangle = pd.Series(index = np.arange(len(bins)-1))
    tmp = angle.groupby(np.digitize(angle.as_units('ms').index.values, bins)-1).mean()
    tmp = tmp.iloc[0:len(bins)-1]
    newangle.loc[tmp.index] = tmp
    newangle.index = pd.Index(bins[0:-1] + np.diff(bins)/2.)

    # BIN SPEED
    position    = position.restrict(ep)
    index         = np.digitize(position.as_units('ms').index.values, bins)
    tmp         = position.groupby(index).mean()
    tmp.index     = bins[np.unique(index)-1]+(bins*1e3)/2
    distance    = np.sqrt(np.power(np.diff(tmp['x']), 2) + np.power(np.diff(tmp['z']), 2))
    speed         = nts.Tsd(t = tmp.index.values[0:-1]+ bin_size/2, d = distance/bin_size)
    speed         = speed.restrict(ep)
    

    tmp = rates.rolling(window=200,win_type='gaussian',center=True,min_periods=1, axis = 0).mean(std=2).values

    imap = Isomap(n_neighbors = 50, n_components = 3, n_jobs = -1).fit_transform(tmp)
#    imap = TSNE(n_components = 3).fit_transform(tmp)
    
    iwak = imap

    H = newangle.values/(2*np.pi)

    HSV = np.vstack((H, np.ones_like(H), np.ones_like(H))).T

    from matplotlib.colors import hsv_to_rgb
    

    RGB = hsv_to_rgb(HSV)

    
    fig = figure()
#    ax = subplot(111)
    ax = subplot(111, projection = '3d')    
#    noaxis(ax)
#    ax.set_aspect(aspect=1)
#    ax.scatter(iwak[~np.isnan(H),0], iwak[~np.isnan(H),1], c = RGB[~np.isnan(H)], marker = '.', alpha = 0.5, zorder = 2, linewidth = 0, s= 40)    
    ax.set_aspect(aspect=1)
    ax.scatter(iwak[~np.isnan(H),0], iwak[~np.isnan(H),1], iwak[~np.isnan(H),2], c = RGB[~np.isnan(H)], s = speed.values*100, alpha = 0.5, linewidth = 0)
    # hsv
#    display_axes = fig.add_axes([0.2,0.45,0.1,0.1], projection='polar')
#    colormap = plt.get_cmap('hsv')
#    norm = mpl.colors.Normalize(0.0, 2*np.pi)
#    xval = np.arange(0, 2*pi, 0.01)
#    yval = np.ones_like(xval)
#    display_axes.scatter(xval, yval, c=xval, s=100, cmap=colormap, norm=norm, linewidths=0, alpha = 0.8)
#    display_axes.set_yticks([])
#    display_axes.set_xticks(np.arange(0, 2*np.pi, np.pi/2))
#    display_axes.grid(False)

    show()

    return iwak

def computeSpeedTuningCurves(spikes, position, ep, bin_size = 0.1, nb_bins = 20, speed_max = 0.4):
    time_bins     = np.arange(position.index[0], position.index[-1]+bin_size*1e6, bin_size*1e6)
    index         = np.digitize(position.index.values, time_bins)
    tmp         = position.groupby(index).mean()
    tmp.index     = time_bins[np.unique(index)-1]+(bin_size*1e6)/2
    distance    = np.sqrt(np.power(np.diff(tmp['x']), 2) + np.power(np.diff(tmp['z']), 2))
    speed         = nts.Tsd(t = tmp.index.values[0:-1]+ bin_size/2, d = distance/bin_size)
    speed         = speed.restrict(ep)
    bins         = np.linspace(0, speed_max, nb_bins)
    idx         = bins[0:-1]+np.diff(bins)/2
    speed_curves = pd.DataFrame(index = idx,columns = np.arange(len(spikes)))
    for k in spikes:
        spks     = spikes[k]
        spks     = spks.restrict(ep)
        speed_spike = speed.realign(spks)
        spike_count, bin_edges = np.histogram(speed_spike, bins)
        occupancy, _ = np.histogram(speed, bins)
        spike_count = spike_count/(occupancy+1)
        speed_curves[k] = spike_count/bin_size

    return speed_curves

"""
Filters
"""

from scipy.signal import butter, lfilter

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

#########################################################
# HELPERS
#########################################################
def writeNeuroscopeEvents(path, ep, name):
    f = open(path, 'w')
    for i in range(len(ep)):
        f.writelines(str(ep.as_units('ms').iloc[i]['start']) + " "+name+" start "+ str(1)+"\n")
        f.writelines(str(ep.as_units('ms').iloc[i]['end']) + " "+name+" end "+ str(1)+"\n")
    f.close()        
    return

def getAllInfos(data_directory, datasets):
    allm = np.unique(["/".join(s.split("/")[0:2]) for s in datasets])
    infos = {}
    for m in allm:
        path = os.path.join(data_directory, m)
        csv_file = list(filter(lambda x: '.csv' in x, os.listdir(path)))[0]
        infos[m.split('/')[1]] = pd.read_csv(os.path.join(path, csv_file), index_col = 0)
    return infos

def computeSpeed(position, ep, bin_size = 0.1):
    time_bins     = np.arange(position.index[0], position.index[-1]+bin_size*1e6, bin_size*1e6)
    index         = np.digitize(position.index.values, time_bins)
    tmp         = position.groupby(index).mean()
    tmp.index     = time_bins[np.unique(index)-1]+(bin_size*1e6)/2
    distance    = np.sqrt(np.power(np.diff(tmp['x']), 2) + np.power(np.diff(tmp['z']), 2))
    speed         = nts.Tsd(t = tmp.index.values[0:-1]+ bin_size/2, d = distance/bin_size)
    speed         = speed.restrict(ep)
    return speed

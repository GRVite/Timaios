import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pylab import *
import neuroseries as nts
import sys
from scipy.signal import find_peaks
import os
import pickle

"""
Notes:
    If the opposite is not indicate it, all the time units are us
"""

class sim():
    """
    This class contains function for the simulation of neural activity
    """
    def __init__(self):
        return
    
    def traingeneration (num_ofneurons, start, end, time_step = 10000000, factor = 1):
        """

        Parameters
        ----------
        num_ofneurons : int
            desired number of neurons to simulate
        start : int
            start time
        end : int
            end time
        time_step : int
            DESCRIPTION. The default is 10000000.
        factor : float, int
            DESCRIPTION. The default is 1.

        Returns
        -------
        spikes : dict
            a dictionary with simulated spike times as values and indices as keys

        """
        #define firing rates for each neuron
        firing_rate = {n:f for n,f in zip ( [*range(num_ofneurons)], 
                                           np.random.randint(20,80,num_ofneurons))}
        spikes = {neuron:emptyList for neuron, emptyList in \
                  zip ([*range(num_ofneurons)],  [[] for i in range(num_ofneurons)])}
        for window in range(start, end, time_step):
            low_edge = window
            high_edge = window + 100000
            # tells you the number of spikes you should expect for each neuron
            firing_rate_scaled = [int(fr*factor) for fr in firing_rate.values()]
            n_spikes = np.random.poisson([*firing_rate_scaled]) 
            # n_spikes = int(n_spikes*factor)
            for n in spikes.keys():
                if n_spikes[n] > 0:
                    # assign to each spike a real timestamp uniformly distributed within the time window
                    array = np.random.uniform(low = low_edge, high = high_edge, 
                                              size = n_spikes[n])
                    spikes[n].extend(array)
        return spikes
    
    
    
    def complextrain(epochs, num_ofneurons):
        """
        This function simulate neural activity during several time epochs of
        activity or innactivity

        Parameters
        ----------
        epochs : a neuroseries.interval_set.IntervalSet
            an IntervalSet with the start and end of a time period and a label
            indicating if the neuron was "active" or "inactive"
        num_ofneurons :  int
            desired number of neurons to simulate

        Returns
        -------
        complex_spikes : dict
            a dictionary with simulated spike times as values and indices as keys

        """
        complex_spikes = {neuron:emptyList for neuron, emptyList in \
                      zip ([*range(num_ofneurons)],  [[] for i in range(num_ofneurons)])}
        for i in epochs.index:
            if epochs.loc[i].label == 'active':
                spikes = sim.traingeneration (num_ofneurons, epochs.loc[i].start, 
                                          epochs.loc[i].end, factor = 1)
            elif epochs.loc[i].label == 'inactive':
                spikes = sim.traingeneration (num_ofneurons, epochs.loc[i].start, 
                                          epochs.loc[i].end, factor = 0.25)
            else: print("wrong label for epochs")
            for n in complex_spikes.keys():
                complex_spikes[n].extend(spikes[n])
        for n in complex_spikes.keys():
            array = np.sort(np.asarray(complex_spikes[n]))
            complex_spikes[n] = nts.Ts(array, time_units = 'us')
        return complex_spikes

class raster:
    def __init__(self):
        return
    
    def gendata(spk, pre, post, lapsos):
        """
        It helps you to generate the data necessary for making a raster of 
        neural activity. This function restricts the neural activity to an interval
        center at a given times.

        Parameters
        ----------
        spk : a dictionary with the spikes times as values and indices as keys.
        pre : time to consider previously to the start time of the stimulus.
        post : time to consider after the start time of the stimulus.
        lapsos : start times, should be a iterable object, as a list.
        Returns
        -------
        spikes_list : a list of spikes restricted to a a given interval and centered
        at the times defined in lapsos.

        """
        spikes_list=[]
        for neuron in spk.keys():
            for i in lapsos:
                interval = nts.IntervalSet(start = i - pre , end = i+post)
                t = spk[neuron].restrict(interval).index.values - i
                spikes_list.append(t)
        return  spikes_list
    
    def gendatans(spikes, epoch, span):
        timep=span/2
        spikes_list=[]
        for i in range(epoch[0],epoch[-1], span):
            interval = nts.IntervalSet(start = i - timep , end = i+timep)
            t = spikes[neuron].restrict(interval).index.values - i
            spikes_list.append(t)
        return  spikes_list
    
    
    

class matrix:
    def __init__(self):
        return
    
    def gendata(spk, pre, post, lapsos, units, binsize):
        """
        It allows you to generate the data for a matrix of neural activity 
        influenced by a given stimulus happenning at a given time.

        Parameters
        ----------
        spk :  a dictionary with the spikes times as values and indices as keys.
        pre : time to consider previously to the start time of the stimulus.
        post : time to consider after the start time of the stimulus.
        lapsos : start times, should be a iterable object, as a list.
        binsize : time bin size.
        Returns
        -------
        spikes_df : a dataframe of neural activity with firing rate per time bin
        """
        bins=arange(pre*-1,post,binsize)
        spikes_df=pd.DataFrame(columns=spk.keys(), index=bins)
        for neuron in spk.keys():
            for i in lapsos:
                interval = nts.IntervalSet(start = i - pre , end = i + post)
                t = spk[neuron].restrict(interval).index.values - i
                bins=arange(pre *-1, post + binsize, binsize)
                t, edges = np.histogram(t, bins)
                spikes_df[neuron]=t
        return  spikes_df
    
    def gendatab(spk, pre_, post, lapsos, binsize, FRbase):
        """
        It allows you to generate the data for a matrix of neural activity 
        showing a change in the firing rate in the presence of a given stimulus

        Parameters
        ----------
        spk :  a dictionary with the spikes times as values and indices as keys.
        pre : time to consider previously to the start time of the stimulus.
        post : time to consider after the start time of the stimulus.
        lapsos : start times, should be a iterable object, as a list.
        binsize : time bin size.
        FRbase : a pd.data frame with the mean firing rate of the neurons during
        the baseline epoch. This dataframe can be obtained with the function
        computeMeanFiringRate(from functions.py)

        Returns
        -------
        spikes_dfb : a dataframe of neural activity binarized. Its values are
        expressed as % of change of firing rate respect to the baseline.

        """
        bins=arange(pre_*-1,post,binsize)
        spikes_dfb=pd.DataFrame(columns=spk.keys(), index=bins)
        for neuron in spk.keys():
            for i in lapsos:
                interval = nts.IntervalSet(start = i - pre_ , end = i + post)
                t = spk[neuron].restrict(interval).index.values - i
                bins=arange(pre_*-1,post+binsize,binsize)
                t, edges = np.histogram(t,bins)
                t = t / 10
                per = t*100/FRbase.iloc[neuron][0]-100
                spikes_dfb[neuron]= per
        return  spikes_dfb

class ephysplots:
    """
    This class contains functions that allow you to easily plot data coming 
    from other functions in this script.
    """
    
    def __init__(self):
        return
    def plot(x):
        # It allows you to show fast a plot from a matplotlib/seaborn function
        # in spyder. Useful when you are writing code and you want to save time.
        plt.figure()
        x
        
    def distplot(df_stim_sns, spikes_keys, label, session, dir2save, 
                 shanks_colors = ['lightblue', 'steelblue', 'royalblue', 'midnightblue']):
        """
        This function help you to create and save distribution plots of the 
        firing rate as % of change for neurons and shanks 

        Parameters
        ----------
        df_stim_sns : pd.DataFrame 
            the output of the funtion pd2seaborn, defined here.
        spikes_keys : 
            the indices of your spikes (0,1,2...)
        label : string 
            this label will be showed in the title and in the named to save the plot
        session : string
            session id, 'A4405-200312'
        dir2save : string
            Address of the directory to save the plots
        shanks_colors : list
            The default is ['lightblue', 'steelblue', 'royalblue', 'midnightblue'].

        Returns
        -------
        None.

        """
        from matplotlib import colors as mcolors
        hsv = []
        for color in shanks_colors:
            hsv.append(mcolors.to_hex(color))
        #palette = sns.xkcd_palette(['lightblue', 'steelblue', 'royalblue', 'midnightblue'])
        palette = sns.color_palette(hsv)
        # sns.set(style="whitegrid")
        # sns.set_context(None)
        fig1 = plt.figure()
        ax1 = fig1.add_subplot(1, 2, 1)
        g = sns.boxenplot (x = 'neurons', y = "firing rate as % change",  
                           hue = 'shanks', palette=palette, data = df_stim_sns, ax= ax1)
        g.set(xticklabels=[*spikes_keys])
        ax2 = fig1.add_subplot(1, 2, 2)
        sns.barplot (x = 'shanks', y = "firing rate as % change",  
                     palette = palette, data = df_stim_sns, ax= ax2)
        plt.suptitle(session + " " + label)
        plt.savefig('{}{}{}'.format(dir2save, '/' + session + "_clustermap_" + label , '.pdf'))


    
    def matrix(pre, post, matrix_data, session, dir2save, label):
        """
        This functions takes the output from matrix.gendata(b) as an input

        Parameters
        ----------
        pre : int
            time to consider previously to the start time of the stimulus.
        post : int
            time to consider after the start time of the stimulus.
        matrix_data : pd.DataFrame
            a dataframe of neural activity binarized. 
        session : string
            session id, 'A4405-200312'
        dir2save : string
            Address of the directory to save the plots
        label : string 
            this label will be showed in the title and in the named to save the plot

        Returns
        -------
        None.

        """
        rango = [*arange(-int(pre/1000/1000),int(post/1000/1000),40)]
        xticklabels = []
        for i in arange(-int(pre/1000/1000),int(post/1000/1000), 10):
            if i in rango:
                xticklabels.append(i)
            else:
                xticklabels.append("")
        fig = plt.figure(figsize = (20, 15))
        ax=fig.add_subplot(111,label="1")
        ax = sns.heatmap(matrix_data.transpose(), cmap= "coolwarm", 
                         xticklabels = xticklabels, center = 0, 
                         cbar_kws={'label': 'firing rate as % change'},
                         square = True)
        ax.set_xlabel("time (s)")
        ax.set_ylabel("neuron")
        ax2=ax.twinx()
        ax2.axvline(ax2.get_xlim()[1]/3, color='white', linewidth=1, zorder=-1)
        ax2.axvline(2*ax2.get_xlim()[1]/3, color='white', linewidth=1, zorder=-1)
        plt.title(session+" {}".format(label))
        plt.show()
        plt.tight_layout()
        plt.savefig('{}{}{}'.format(dir2save,  '/' + session + "_matrixb_" + label , '.pdf'))    
        
    def raster(raster_list,  stimduration, neurons_sel, session, dir2save,
             intensity_label = "High intensity", units = 's', binsize = 1, \
                 ylabel = "Firing Rate", linesize = 0.5, \
                     colorstim = "lightcyan",  colorctrl = "tan"):
        """

        Parameters
        ----------
        raster_list : list
            the output from raster.gendata
        stimduration : int
            duration of the stimulus in seconds
        neurons_sel : list
            indices of the neurons
        session : string
            session id, 'A4405-200312'
        dir2save : string
            address of the directory to save the plots
        intensity_label : string
            the default is "High intensity".
        units : string
            it can be 'ms' or 's'. The default is 's'.
        binsize : int
            the default is 1.
        ylabel : string
            the default is "Firing Rate".
        linesize : float
            the default is 0.5.
        colorstim : string
            the default is "lightcyan".
        colorctrl : string
            the default is "tan".

        Returns
        -------
        None.

        """
        if units == 'ms':
            scale = 1000
        elif units == 's':
            scale = 1000000
        else:
            print("wrong units input")
        # Scale the raster data based on the units selected
        raster_list= [i/scale for i in raster_list]    
        # Create an array for your histogram
        array_hist = np.concatenate(raster_list).ravel()
        # Generate the bins for this array
        nbins = int(array_hist[-1]*2/binsize)
        fig, (ax1,ax2) = plt.subplots(2, 1, sharex = True, figsize = [16,12])
        ax1.eventplot(raster_list, linelengths = linesize, color='black')
        left, bottom, height = (0, 0.5, len(raster_list))
        rect = plt.Rectangle((left, bottom), stimduration, height,
                             facecolor = colorstim, alpha = 0.5)
        ax1.add_patch(rect)
        ax1.set_ylabel('Neurons')
        ax1.set_yticks([*range(len(neurons_sel))])
        ax1.set_yticklabels( {v:k for k,v in enumerate(neurons_sel)} ) 
        ax1.legend(["ChR2"])
        ax1.set_frame_on(False)
        data, edges = np.histogram(array_hist, nbins)
        ax2.hist(array_hist, bins = nbins, color = colorctrl)
        ax2.plot([0, stimduration],[data.max(), data.max()], linewidth = 7, 
                 color = colorstim )
        ax2.set_xlabel("{}{}".format('Time (',units+')'))
        ax2.set_ylabel(ylabel)
        ax2.set_frame_on(False)
        plt.show()
        plt.suptitle(session + " " + intensity_label)
        plt.savefig('{}{}{}'.format(dir2save,  '/' + session + "_raster_neuronselected", '.pdf'))   

##############################################################################
##############################################################################
        
# Functions, various
        
##############################################################################
##############################################################################
try:    
    def accessgoogle(SAMPLE_SPREADSHEET_ID_input, SAMPLE_RANGE_NAME):
        """
        This function allows you to access a google document from your script.
        was copied from https://developers.google.com/sheets/api/quickstart/python
    
        Parameters
        ----------
        SAMPLE_SPREADSHEET_ID_input: the id of your google sheet, 
            can be obtained from its link example: For this link,
       https://docs.google.com/spreadsheets/d/1DiJMx6G9IhU_X_QY6NTWbqBWh5CvvLsoQVdo4IN0KXc/edit#gid=148245886
       the id is 1DiJMx6G9IhU_X_QY6NTWbqBWh5CvvLsoQVdo4IN0KXc
       
        SAMPLE_RANGE_NAME: The columns and cells you want to access, example:
             'A1:AA100'        
             
        Returns
        -------
        values_input : a list of lists, each list represents a raw of the spread sheet.
    
        """
        from googleapiclient.discovery import build
        from google_auth_oauthlib.flow import InstalledAppFlow,Flow
        from google.auth.transport.requests import Request
        global values_input, service
        creds = None
        if os.path.exists('token.pickle'):
            with open('token.pickle', 'rb') as token:
                creds = pickle.load(token)
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                flow = InstalledAppFlow.from_client_secrets_file(
                    'credentials.json', SCOPES) # here enter the name of your downloaded JSON file
                creds = flow.run_local_server(port=0)
            with open('token.pickle', 'wb') as token:
                pickle.dump(creds, token)
    
        service = build('sheets', 'v4', credentials=creds)
    
        # Call the Sheets API
        sheet = service.spreadsheets()
        result_input = sheet.values().get(spreadsheetId=SAMPLE_SPREADSHEET_ID_input,
                                    range=SAMPLE_RANGE_NAME).execute()
        values_input = result_input.get('values', [])
    
        if not values_input and not values_expansion:
            print('No data found')
        return values_input
except ModuleNotFoundError as die:
    print(die)
    pass


def optoeps(ttl_start,ttl_end, height = 1):
    """
    This function help you to get the start and end times of the diferent 
    light intensities of the stimulation. It achieves this by looking for
    big differences in time among the TTL pulses. 

    Parameters
    ----------
    ttl_start : neuroseries.time_series.Ts
        start time of all the ttl
    ttl_end : neuroseries.time_series.Ts
        DESCRIPTION.
    height : float, int
        Height of the peak, the time difference in seconds between 2 TTL pulses.
        The default is 1.

    Returns
    -------
    stim_ep : 
        an interval set with the beginning and end of the different 
    intensities of the stimulation

    """
    t, _ = find_peaks(np.diff(ttl_start.as_units('s').index), height)
    stim_ep = np.sort(np.hstack(([ttl_start.index[0]], ttl_start.index[t], 
                                 ttl_start.index[t+1], ttl_end.index[-1])))
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

def pd2seaborn(df_stim, spikes_keys, shanks):
    """

    Parameters
    ----------
    df_stim : data frame of neural activity during the stimulation epoch
    with firing rate as % of change respect to the baseline 

    Returns
    -------
    sn_df : a dataframe with the format easy to plot using seaborn

    """
    sn_data = []
    for i in spikes_keys: 
        sn_data.append(df_stim[i].values.flatten())
    sn_data = np.concatenate(sn_data).ravel()
    stim = df_stim.index
    sn_neurons = []
    for i in spikes_keys:
        for j in range(len(stim)):
            sn_neurons.append(int(i))
    sn_shanks = []
    for i in shanks:
        for j in range(len(stim)):
            sn_shanks.append(int(i))
    sn_time = []
    for i in spikes_keys:
        sn_time.append(stim)
    sn_time = np.concatenate(sn_time).ravel()
    sn_df = pd.DataFrame(np.stack((sn_data, sn_time, sn_neurons, sn_shanks), axis =1), columns = ["firing rate as % change","time", "neurons", "shanks"])
    return sn_df
  
        
"""
WORKING


import numpy as np 
import matplotlib.pyplot as plt
xa = 0
xb = 10

u = np.ones((10,10))
function = -3*x2 + 15
ranget = [0,10]
xlabel = "time"
ylabel = "current"
plot_streamplot(ranget, u, v, xlabel, ylabel, figtitle = "Streamplot for a dif equation")

def plot_streamplot(ranget, u, v, xlabel, ylabel, figtitle=None, color = 'turquoise'):

    Show a stream plot for a linear ordinary differential equation with 
    state vector x=[x1,x2] in axis ax.

    Args:
        ranget = range for values of x, representing time
        u = 
        v = differential ecuation

    Returns:
      nothing, but shows a figure
  
    
    # sample 20 x 20 grid uniformly to get x1 and x2
    grid = np.arange(ranget[0], ranget[1], 1)
    x, y = np.meshgrid(grid, grid)
    # make a colormap
    # magnitude = np.sqrt(x1dot ** 2 + x2dot ** 2)
    # color = 2 * np.log1p(magnitude) #Avoid taking log of zero   
    # plot
    # plt.sca(ax)
    plt.figure()
    plt.streamplot(x, y, u, v, color=color, 
                   linewidth=1, cmap=plt.cm.cividis, density=2, arrowstyle='->', arrowsize=1.5)
    
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    """
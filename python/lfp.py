import numpy as np

def downsample(tsd, up, down):
    import scipy.signal
    import neuroseries as nts
    dtsd = scipy.signal.resample_poly(tsd.values, up, down)
    dt = tsd.as_units('s').index.values[np.arange(0, tsd.shape[0], down)]
    if len(tsd.shape) == 1:		
        return nts.Tsd(dt, dtsd, time_units = 's')
    elif len(tsd.shape) == 2:
        return nts.TsdFrame(dt, dtsd, time_units = 's', columns = list(tsd.columns))

def loadLFP(path, n_channels=32, channel=12, frequency=1250.0, precision='int16'):
	import neuroseries as nts
	if type(channel) is not list:
		f = open(path, 'rb')
		startoffile = f.seek(0, 0)
		endoffile = f.seek(0, 2)
		bytes_size = 2		
		n_samples = int((endoffile-startoffile)/n_channels/bytes_size)
		duration = n_samples/frequency
		interval = 1/frequency
		f.close()
		with open(path, 'rb') as f:
			data = np.fromfile(f, np.int16).reshape((n_samples, n_channels))[:,channel]
		timestep = np.arange(0, len(data))/frequency
		return nts.Tsd(timestep, data, time_units = 's')
	elif type(channel) is list:
		f = open(path, 'rb')
		startoffile = f.seek(0, 0)
		endoffile = f.seek(0, 2)
		bytes_size = 2
		
		n_samples = int((endoffile-startoffile)/n_channels/bytes_size)
		duration = n_samples/frequency
		f.close()
		with open(path, 'rb') as f:
			data = np.fromfile(f, np.int16).reshape((n_samples, n_channels))[:,channel]
		timestep = np.arange(0, len(data))/frequency
		return nts.TsdFrame(timestep, data, time_units = 's')

def loadBunch_Of_LFP(path,  start, stop, n_channels=32, channel=64, frequency=1250.0, precision='int16'):
	import neuroseries as nts	
	bytes_size = 2		
	start_index = int(start*frequency*n_channels*bytes_size)
	stop_index = int(stop*frequency*n_channels*bytes_size)
	fp = np.memmap(path, np.int16, 'r', start_index, shape = (stop_index - start_index)//bytes_size)
	data = np.array(fp).reshape(len(fp)//n_channels, n_channels)

	if type(channel) is not list:
		timestep = np.arange(0, len(data))/frequency
		return nts.Tsd(timestep, data[:,channel], time_units = 's')
	elif type(channel) is list:
		timestep = np.arange(0, len(data))/frequency		
		return nts.TsdFrame(timestep, data[:,channel], time_units = 's')
    
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

import csv, math, os, re, statistics, sys
import numpy as np
from scipy.signal import find_peaks
from scipy.stats import kurtosis, skew, iqr


# Code from WatchAuth

import csv, math, os, re, shutil, sys
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import butter, lfilter, freqz, lfilter_zi, filtfilt


#configs
ORDER = 6
CUTOFF = 3.667
FILTER_INDICES = [0,1,2,4,5,6,8,9,10,11,12,13,14] #  which channels need the filter


def butter_lowpass(cutoff, r, order):
    nyq = 0.5 * r
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype = 'low', analog = False)
    return (b, a)

def butter_lowpass_filter(d, cutoff, r, order):
    b, a = butter_lowpass(cutoff, r, order)
    
    zi = lfilter_zi(b, a)
    
    z, _ = lfilter(b, a, d, zi=zi*d[0])
    
    z2, _ = lfilter(b, a, z, zi=zi*z[0])
    
    y = filtfilt(b, a, d)
    
    return y

def filter(data, filter_indices, order, cutoff):
    
    n = data.shape[1]
    
    timespan = 4
    times = np.linspace(-4, 0, num=n)
    
    t = np.linspace(times[0], times[len(times) - 1], n, endpoint = False) #evenly spaced time intervals
    r = int(n / (times[len(times) - 1] - times[0])) #sample rate, Hz
    
    
    new_data = data # .copy()
    
    for i in range(data.shape[0]):
        for j in filter_indices:
            filtered = butter_lowpass_filter(data[i,:,j], cutoff, r, order)
            #plt.plot(data[i,:,j])
            #plt.plot(filtered)
            #plt.show()
            new_data[i,:,j] = filtered

    return new_data



def feature_min(g_data):
    return np.amin(g_data, axis=0)

def feature_max(g_data):	
    return np.amax(g_data, axis=0)

def feature_mean(g_data):
	return np.mean(g_data, axis=0)

def feature_med(g_data):
	return np.median(g_data, axis=0)
	
def feature_stdev(g_data):
	return np.std(g_data, axis=0)

def feature_var(g_data):
	return np.var(g_data, axis=0)

def feature_iqr(g_data):
    return iqr(g_data, axis=0)

def feature_kurt(g_data):
	return kurtosis(g_data, axis=0)

def feature_skew(g_data):
	return skew(g_data, axis=0)

def feature_pkcount(g_data, threshold):
    channels = g_data.shape[1]
    peaks = []
    for i in range(channels):
        datum = g_data[:, i]
        peaks.append(len(find_peaks(datum, prominence = threshold)[0]))
    return np.array(peaks)

def feature_velo_disp(g_data):
    f = []
    indices= [[0,1,2], [4,5,6], [12,13,14]]
    n = g_data.shape[1] - 1
    
    for a,b,c in indices:
        vx = [0]
        dx = [0]
        vy = [0]
        dy = [0]
        vz = [0]
        dz = [0]
        d = [0]
        dt = float(4.0 / n) #sample interval - based on a time slice of 4 seconds
        for j in range(n):
            vx.append(vx[j] + (g_data[j][a] + g_data[j + 1][a]) / 2 * dt / 10)
            dx.append(dx[j] + vx[j + 1] * dt / 10)
            vy.append(vy[j] + (g_data[j][b] + g_data[j + 1][b]) / 2 * dt / 10)
            dy.append(dy[j] + vy[j + 1] * dt / 10)
            vz.append(vz[j] + (g_data[j][c] + g_data[j + 1][c]) / 2 * dt / 10)
            dz.append(dz[j] + vz[j + 1] * dt / 10)
            d.append(math.sqrt(dx[j+1] **2 + dy[j+1] **2 + dz[j+1] **2 ))
        vx.pop(0)
        vy.pop(0)
        vz.pop(0)

        if False:
            f_names.append(sensor + '-x-velomean')
            f_names.append(sensor + '-y-velomean')
            f_names.append(sensor + '-z-velomean')
            f_names.append(sensor + '-x-velomax')
            f_names.append(sensor + '-y-velomax')
            f_names.append(sensor + '-z-velomax')
            f_names.append(sensor + '-x-disp')
            f_names.append(sensor + '-y-disp')
            f_names.append(sensor + '-z-disp')
            f_names.append(sensor + '-disptotal')

        f.append(sum(vx) / len(vx))
        f.append(sum(vy) / len(vy))
        f.append(sum(vz) / len(vz))
        f.append(max(vx, key = abs))
        f.append(max(vy, key = abs))
        f.append(max(vz, key = abs))
        f.append(dx[len(dx) - 1])
        f.append(dy[len(dy) - 1])
        f.append(dz[len(dz) - 1])
        f.append(d[len(d) - 1])
    return np.array(f)

def extract_features(g_data):   
	f_data = []
	f_data.append(feature_min(g_data))
	f_data.append(feature_max(g_data))
	f_data.append(feature_mean(g_data))
	f_data.append(feature_med(g_data))
	f_data.append(feature_stdev(g_data))
	f_data.append(feature_var(g_data))
	f_data.append(feature_iqr(g_data))
	f_data.append(feature_kurt( g_data))
	f_data.append(feature_skew(g_data))
	f_data.append(feature_pkcount(g_data, 0.5))
	f_data.append(feature_velo_disp(g_data))
	return f_data

def featurize(data, filter_indices=FILTER_INDICES):
    
    filtered_data = filter(data, filter_indices, ORDER, CUTOFF)
    
    new_data = np.zeros((filtered_data.shape[0], 200, 19))
    
    
    new_data[:,:,:16] = filtered_data
    new_data[:,:,16] = (filtered_data[:,:,0]**2 + filtered_data[:,:,1]**2 + filtered_data[:,:,2]**2)**0.5
    new_data[:,:,17] = (filtered_data[:,:,4]**2 + filtered_data[:,:,5]**2 + filtered_data[:,:,6]**2)**0.5
    new_data[:,:,18] = (filtered_data[:,:,12]**2 + filtered_data[:,:,13]**2 + filtered_data[:,:,14]**2)**0.5
    
    features = []

    for sample in new_data:
        feature = extract_features(sample)
        features.append(np.concatenate(feature).flatten())
    
    return np.array(features)



def featurize_acc_gyr(data, filter_indices=FILTER_INDICES):
    
    filtered_data = filter(data, filter_indices, ORDER, CUTOFF)
    
    new_data = np.zeros((filtered_data.shape[0], 200, 10))
    
    
    new_data[:,:,:8] = filtered_data
    new_data[:,:,8] = (filtered_data[:,:,0]**2 + filtered_data[:,:,1]**2 + filtered_data[:,:,2]**2)**0.5
    new_data[:,:,9] = (filtered_data[:,:,4]**2 + filtered_data[:,:,5]**2 + filtered_data[:,:,6]**2)**0.5
    
    features = []
    
    print(new_data.shape)

    for sample in new_data:
        feature = extract_features(sample)
        features.append(np.concatenate(feature).flatten())
    
    return np.array(features)


if __name__ == "__main__":

    data = [[]]

    features = []

    for sample in data:
        feature = extract_features(sample)
        features.append(np.concatenate(feature).flatten())
    
    
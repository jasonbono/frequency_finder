'''
Author: Jason Bono (jbono@fnal.gov)
Date: December 5, 2018
    
Purpose: Library for high precision frequency extraction from NMR probe data

Description: Contains various tools for frequency extraction, including a new technique called "Cosecant Transform"
    
'''

import numpy as np



#The Cosecant Transform divides the entire signal by a pure sine wave with a given frequency and phase. When that frequency and phase match, the transformed signal reduces to the envelope of the signal. Small perturbations in frequency/phase cause the transformed values to explode, rapidly decreasing the similarity to envelope, and rapidly increasing the RMS.
#- This function performs the on the passed signal
#- Call with sample_rate in Hz
#- Call with frequency in Hz and phase in rad
#- Returns a (mutable) list containing the transformed data and its varience
def cosecant_transform(data,sample_rate,frequency,phase):
    # To avoid dividing by small numbers, define a clip value
    clip_value = 0.05
    # Convert Hz to rad/s
    frequency = frequency*2.0*np.pi
    # Get the number of points
    data_size = data.size
    # Get the amount of time elapsed from point to point
    time_scale = 1.0/sample_rate
    # Loop over the data, and perform the transform
    new_data = np.empty(0)
    times = np.empty(0)
    for i in range(data_size):
        time = time_scale*i
        sine_val = np.sin(time*frequency + phase)
        if (abs(sine_val) > clip_value):
            new_data = np.append(new_data, data[i]/sine_val)
            times = np.append(times, time)
    return [new_data, np.var(new_data),times]


#This function does a 20x30 scan of the cosecant_transform
#This function performs the on the passed signal
#Call with sample_rate in Hz
#Call with frequency/f_range in Hz and phase/p_range in rad
def scan_cosecant_transform(data,sample_rate,init_freq,init_phase,f_range,p_range):
    
    # Set the range and step size of the frequency scan
    low_f = init_freq - f_range/2.0
    high_f = init_freq + f_range/2.0
    n_f = 20.0
    f_width = f_range/n_f
    
    # Set the range and step size of the phase scan
    low_p = init_phase - p_range/2.0
    high_p = init_phase + p_range/2.0
    n_p = 30.0
    p_width = p_range/n_p
    
    lowest_var = float('inf')
    
    # Loop over frequency
    for f in np.arange(low_f, high_f, f_width):
        # Loop over phase
        for p in np.arange(low_p, high_p, p_width):
            result = cosecant_transform(data,sample_rate,f,p)
            if (result[1] < lowest_var):
                lowest_var = result[1]
                lowest_var_p = p
                lowest_var_f = f
#    print ("done")
    return [lowest_var, lowest_var_f, lowest_var_p]

# A function that calculates average sample rate
def srate(time):
    # Seperate odd/even index
    odd_time = time[1::2]
    even_time = time[::2]
    # Removal of last point ensures that even_time wont go out of index
    n_points = odd_time.size - 1
    diff_time = abs(odd_time[1:n_points] - even_time[1:n_points])
    sample_rate = 1.0/np.mean(diff_time)
    #if the time readings are in s, the sample rate is in Hz
    return sample_rate


# A wrapper function for numpy's fft.fftfreq
# Performs a dfft and returns the peak frequency
def fft_peak(signal,srate):
    from scipy import fftpack
    # The time step
    time_step = 1.0/srate
    
    # The FFT of the signal
    signal_fft = fftpack.fft(signal)

    # The power of signal_fft (signal_fft is complex dtype)
    power = np.abs(signal_fft)

    # The power's associated frequencies
    sample_freq = fftpack.fftfreq(signal.size, d=time_step)
    
    # Peak frequency
    pos_mask = np.where(sample_freq > 0)
    freqs = sample_freq[pos_mask]
    power = power[pos_mask]
    peak_freq = freqs[power.argmax()]
    
    return [freqs,power,peak_freq]

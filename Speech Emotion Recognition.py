#!/usr/bin/env python
# coding: utf-8

# # Speech Emotion Recognition

# In[52]:


import os 
from os.path import isdir, join 
from pathlib import Path 
import pandas as pd 
import seaborn as sns
 
# Math 
import numpy as np 
from scipy.fftpack import fft 
from scipy import signal 
from scipy.io import wavfile 

 
from sklearn.decomposition import PCA 
 
# Visualization 
import matplotlib.pyplot as plt 
import seaborn as sns 
import IPython.display as ipd 

 
import plotly.offline as py 
py.init_notebook_mode(connected=True) 
import plotly.graph_objs as go 
import plotly.tools as tls 
import pandas as pd 


# In[17]:


def log_specgram(audio, sample_rate, window_size=20, 
                 step_size=10, eps=1e-10):
    nperseg = int(round(window_size * sample_rate / 1e3)) 
    noverlap = int(round(step_size * sample_rate / 1e3)) 
    freqs, times, spec = signal.spectrogram(audio, 
                                    fs=sample_rate, 
                                    window='hann', 
                                    nperseg=nperseg, 
                                    noverlap=noverlap, 
                                    detrend=False) 
    return freqs, times, np.log(spec.T.astype(np.float32) + eps) 

# Define the directory path
base_path = "C:/Users/ranji/Desktop/python/project/Code Alpha project/Speech Emotion Recognition/Tess/"

# Define the emotions/folders
emotions = ['OAF_angry', 'OAF_disgust', 'OAF_Fear', 'OAF_happy', 'OAF_neutral', 'OAF_Sad','OAF_Pleasant_surprise']

# Process each folder
for emotion in emotions:
    # Get the full path of the folder
    folder_path = os.path.join(base_path, emotion)
    
    # List all files in the folder
    files = [f for f in os.listdir(folder_path) if f.endswith('.wav')]
    
    # Process each WAV file in the folder
    for file in files:
        # Read the WAV file
        file_path = os.path.join(folder_path, file)
        sample_rate, samples = wavfile.read(file_path)
        
        # Compute spectrogram
        freqs, times, spectrogram = log_specgram(samples, sample_rate)
        
        # Perform further processing as needed
        # For example, you can save the spectrogram or perform analysis
        
        # Print some information (you can replace this with your desired processing)
        print(f"Processed {file} from {emotion} folder.")


# In[24]:


# Plot the raw wave
plt.figure(figsize=(12, 4))
plt.plot(np.linspace(0, len(samples) / sample_rate, len(samples)), samples)
plt.title('Raw wave of ' + x)
plt.ylabel('Amplitude')
plt.xlabel('Time [s]')
plt.show()


# In[26]:


ax2 = fig.add_subplot(212) 
ax2.imshow(spectrogram.T, aspect='auto', origin='lower',  
           extent=[times.min(), times.max(), freqs.min(), freqs.max()]) 
ax2.set_yticks(freqs[::16]) 
ax2.set_xticks(times[::16]) 
ax2.set_title('Spectrogram of ' + filename) 
ax2.set_ylabel('Freqs in Hz') 
ax2.set_xlabel('Seconds') 
mean = np.mean(spectrogram, axis=0) 
std = np.std(spectrogram, axis=0) 
spectrogram = (spectrogram - mean) / std 
a=["0a7c2a8d_nohash_0.wav","0a7c2a8d_nohash_1.wav","0a7c2a8d_nohash_2.wav","0a7c2a8d_nohash_3.wav","0a7c2a8d_nohash_4.wav"] 
d={ 
   0: "neutral", 
    1: "calm", 
    2: "happy", 
    3: "sad", 
    4: "angry", 
    5: "fearful", 
    6: "disgust",  
    7: "surprised" 
} 


# In[32]:


# Define the range of samples you want to plot
start_index = 0  # Start index of the range
end_index = 1000  # End index of the range

# Extract the samples in the defined range
samples_cut = samples[start_index:end_index]

# Compute spectrogram
freqs, times, spectrogram = signal.spectrogram(samples, fs=sample_rate)

# Define the range of spectrogram you want to plot
start_index_spec = 0  # Start index of the range
end_index_spec = 100  # End index of the range

# Extract the spectrogram in the defined range
spectrogram_cut = spectrogram[:, start_index_spec:end_index_spec]

# Plot raw wave and spectrogram
fig = plt.figure(figsize=(14, 8))

# Raw wave
ax1 = fig.add_subplot(211)
ax1.set_title('Raw wave of ' + os.path.basename(base_path))
ax1.set_ylabel('Amplitude')
ax1.plot(samples_cut)

# Spectrogram
ax2 = fig.add_subplot(212)
ax2.set_title('Spectrogram of ' + os.path.basename(base_path))
ax2.set_ylabel('Frequencies * 0.1')
ax2.set_xlabel('Samples')
ax2.imshow(spectrogram_cut.T, aspect='auto', origin='lower',  
           extent=[times[start_index_spec], times[end_index_spec], freqs.min(), freqs.max()])
ax2.set_yticks(freqs[::16])
ax2.set_xticks(times[start_index_spec:end_index_spec:16])
plt.show()


# In[33]:


xcoords = [0.025, 0.11, 0.23, 0.49] 
for xc in xcoords: 
    ax1.axvline(x=xc*16000, c='r') 
    ax2.axvline(x=xc, c='r') 
def custom_fft(y, fs): 
    T = 1.0 / fs 
    N = y.shape[0] 
    yf = fft(y) 
    xf = np.linspace(0.0, 1.0/(2.0*T), N//2) 
    vals = 2.0/N * np.abs(yf[0:N//2])  # FFT is simmetrical, so we take just the first half 
    # FFT is also complex, to we take just the real part (abs) 
    return xf, vals 
xf, vals = custom_fft(samples, sample_rate) 
plt.figure(figsize=(12, 4)) 
plt.title('FFT of recording sampled with ' + str(sample_rate) + ' Hz') 
plt.plot(xf, vals) 
plt.xlabel('Frequency') 
plt.grid() 
plt.show() 
print("Emotion of the recording is ",d[0]) 


# In[34]:


xf,vals = custom_fft(samples, sample_rate)
plt.figure(figsize=(12,4))
plt.title('FFt of recording sampled with' + str(sample_rate) + 'Hz')
plt.plot(xf,vals)
plt.xlabel('Frequncy')
plt.grid()
plt.show()


# In[45]:


samples_cut = samples[4000:13000]
ipd.Audio(samples_cut,rate=sample_rate)


# In[46]:


ipd.Audio(samples,rate=sample_rate)


# In[48]:


def custom_fft(y,fs):
    T = 1.0 / fs
    N = y.shape[0]
    yf = fft(y)
    xf = np.linspace(0.0,1.0 / (2.0*T), N//2) 
    vals = 2.0/N * np.abs(yf[0:N//2]) #FFT is simmetrical, so we take just the first half
    #FFT is also complex,to we take just the real part(abs)
    return xf, vals


# In[50]:


ipd.Audio(samples, rate=sample_rate)


# In[51]:


print("Emotion of the recording is ", d[0])


# In[ ]:







#from scipy.fft import fft, fftfreq
import numpy as np
from numpy.fft import fft, fftfreq, rfft, rfftfreq, fftshift

import matplotlib

from matplotlib import pyplot as plt

matplotlib.use('TkAgg') #----> Specify the backend

dp  = np.dtype('d')        # double precision
dpc = np.dtype(np.cdouble) # double precision complex
i4  = np.dtype('i4') # integer 4
i8  = np.dtype('i8') # integer 8

# Frequency (Hz)
freq = 5

#print("np.arange(5) = ", np.arange(5))

Ns = 11 # number of samples

Fs = 100 # sampling frequency
dt = 1./Fs # time discretisation
time = np.arange(Ns)*dt # time sampling

print("Number of samples: ", Ns)
print("len(time) = ", len(time))
print("")

# f = [0, 1, ...,   n/2-1,     -n/2, ..., -1] / (dt*n)   if n is even ==> n frequencies
# f = [0, 1, ..., (n-1)/2, -(n-1)/2, ..., -1] / (dt*n)   if n is odd ==> n frequencies
freq_theory = np.zeros(Ns, dp)
if (Ns%2==0): # even
    fmt1 = np.arange(0, int(Ns/2)-1 +1, 1) # +1 because arange does not contain upper limit
    fmt2 = np.arange(-int(Ns/2), -1 +1, 1) # +1 because arange does not contain upper limit
else: # odd
    fmt1 = np.arange(0, int((Ns-1)/2) +1, 1) # +1 because arange does not contain upper limit
    fmt2 = np.arange( -int((Ns-1)/2) , -1+1, 1) # +1 because arange does not contain upper limit

freq_theory = np.sort( np.concatenate((fmt1, fmt2), axis=0)/(Ns*dt) )
print("freq_theory (sorted) = ", freq_theory)
    
#time = np.linspace(0, 1./freq, Ns, endpoint=False)
#dt = time[1]-time[0]
#Fs=1./dt

fNyquist = Fs/2.
print("")
print("fNyquist = ", fNyquist)
#print("time = ",time)
#print("dt = ", dt)

# 2pi because np.sin takes radians
y = np.sin((2.*np.pi*freq) * time)#**2.
#yfull = np.sin((2.*np.pi*freq) * timefull)#**2.

y_ave = np.average(y)
#yfull_ave = np.average(yfull)

print("")
print("y_ave = ", y_ave)
#print("yfull_ave = ", yfull_ave)

#yf = rfft(y)
#frequencies = rfftfreq(Ns, dt)
yf = fft(y)
frequencies = fftfreq(Ns, dt)
    
print("")
scale = 2.*np.ones_like(frequencies) # scale rfft components by a factor of 2
scale[0] = 1. #  the DC component is not scaled (bacause only 1 dc component)

if (frequencies.size%2 == 0):
    print("even number of frequencies ==> there is a Nyquist component")
    print("-----------------------------------------------------------")
    idx_nyq = int(Ns/2) # this is for array frequencies (i.e. not sorted)
    scale[idx_nyq] = 1. # ...then it is not scaled (bacause it shows up only once)
else:
    print("odd number of frequencies ==> no Nyquist component")
    print("--------------------------------------------------")

print("")
print("frequencies = ", frequencies)
#print("frequencies[Ns//2] (Nyquist freq.) = ", frequencies[Ns//2])
#print("frequencies[idx_nyq] (Nyquist freq.) = ", frequencies[idx_nyq])
#print("Ns, Ns//2 = ", Ns, Ns//2)

# compute frequency associated
# with coefficients
#print("")
#print("freqs[idx] = ", frequencies[idx])

plt.plot(time, y)
plt.show()

abs_yf =  np.abs(yf)
plt.plot(frequencies, abs_yf, 'bs')
plt.show()

idx_arr =  np.argsort(frequencies)
freq_sorted = frequencies[idx_arr]
abs_yf_sorted = abs_yf[idx_arr]
print("freq_sorted = ", freq_sorted)

data_out = np.column_stack([ freq_sorted, abs_yf_sorted ])
datafile_path = "./debug_fft_python.dat"
np.savetxt(datafile_path , data_out, fmt=['%21.11e','%21.11e'])









# ntimes = 1
# duration = ntimes*1./freq
# print("")
# print("duration = ", duration)

#time = np.linspace(0, duration, Ns, endpoint=False)
#timefull = np.linspace(0, duration, Ns, endpoint=True)

# Note: sample_rate = 1/time_step

#time_step = duration/Ns
#print("")
#print("time_step = ", time_step)

#time_step2 = time[1]-time[0]
#print("time_step2 = ", time_step2)

#print("")
#print("sampling rate = ", 1./time_step)

#fNyq = 1./(2.*time_step)
#print("fNyq = ", fNyq)

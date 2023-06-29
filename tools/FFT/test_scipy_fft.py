

from scipy.fft import fft, fftfreq
import numpy as np

import matplotlib

from matplotlib import pyplot as plt

matplotlib.use('TkAgg') #----> Specify the backend

#sample_rate = 500  # Hertz
duration = 2.*np.pi  # Seconds

# Frequency (Hz)
freq = 2

nsample = 256

#x = np.linspace(0, duration, nsample, endpoint=False)
#xfull = np.linspace(0, duration, nsample, endpoint=True)

x = np.linspace(0, duration, nsample, endpoint=False)
xfull = np.linspace(0, duration, nsample, endpoint=True)

sample_rate = duration/nsample
print("sample_rate = ", sample_rate)

sample_rate2 = x[1]-x[0]
print("sample_rate2 = ", sample_rate2)


#frequencies = x * freq
#frequencies_full = xfull * freq

# 2pi because np.sin takes radians
y = np.sin((2 * np.pi) * x)**2.
yfull = np.sin((2 * np.pi) * xfull)**2.

y_ave = np.average(y)
yfull_ave = np.average(yfull)

print("y_ave = ", y_ave)
print("yfull_ave = ", yfull_ave)

yf = fft(y)
xf = fftfreq(nsample, 1 / sample_rate)

print("yf[0]/nsample = ", yf[0]/(nsample))

print("For sin(x)^2 over [0,2p], exact average is: np.pi/(2*pi) = ", np.pi/(2*np.pi))

plt.plot(x, y)
plt.show()

plt.plot(xf, np.abs(yf))
plt.show()

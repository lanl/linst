
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
import numpy as np

dp  = np.dtype('d')        # double precision
dpc = np.dtype(np.cdouble) # double precision complex

i4  = np.dtype('i4') # integer 4
i8  = np.dtype('i8') # integer 8

# Number of sample points
Ns = 21

# wavenumber
kx = 3.1234 
Lamx = 2.*np.pi/kx 

# By analogy with the time frequency, the space frequency is fx = 1./Lamx
fx = 1./Lamx
print("")
print("Space frequency is fx = ", fx)
print("")

# samples
x = np.linspace(0.0, Lamx, Ns, endpoint=False)
x2 = np.linspace(0.0, Lamx, Ns, endpoint=True)

dx = x[1]-x[0]

Fs = 1/dx
print("")
print("Sampling frequency is Fs = ", Fs)
print("")

fNyquist = Fs/2.
print("fNyquist = ", fNyquist)
print("")

print_this = False

if (print_this):
    print("")
    print("Lamx = ", Lamx)
    print("")
    print("x = ", x)
    print("")
    print("x2 = ", x2)

    print("")
    print("Ns, len(x), len(x2) = ", Ns, len(x), len(x2))

# Create signal
y  = np.sin(kx*x) #+ 0.5*np.sin(80.0 * 2.0*np.pi*x)

# We know period of sin^n(x) is 2π when n is odd integer and π when n is even integer
# Note sine(x)^3 = 0.25*( 3*sine(x) - sine(3*x) )

# To check that normalization (fft(y)/Ns) is correct, use y = y**2. and chech that the average in the spectrum = 0.5
take_cube=False
if (take_cube):
    y = y**3.
    freq3 = 3.*fx
    print("")
    print("Frequency f = %21.11e should also be present" % freq3)
    print("")

# Compute fft and normalize it
yf = fft(y)/Ns
frequencies = fftfreq(Ns, dx)

print("")
print("frequencies = ", frequencies)
print("")

# dc and Nyquist modes come always by themselves so do not multiply by 2
scale = 2.*np.ones_like(frequencies)
scale[0] = 1. # do not multiply by 2 DC component

# Note: operator // is floor division in python
if ( Ns % 2 == 0 ):
    print("Ns is even")
    print("Nyquist mode is present")
    print("f_Nyq = ", frequencies[Ns//2])
    print("Ns, Ns//2 = ", Ns, Ns//2)
    
    scale[Ns//2] = 1. # do not multiply by 2 Nyquist component because it comes alone

    idx_end = Ns//2

    # Not including dc mode
    freq_pos = frequencies[1:Ns//2]
    # With dc mode
    freq_dc_pos = frequencies[0:Ns//2]
    # Not including Nyquist mode
    freq_neg = frequencies[Ns//2+1:]
    #print("freq_pos = ", freq_pos)
    #print("freq_neg = ", freq_neg)
    
else:
    print("Ns is odd")
    print("Nyquist mode is absent")
    print("Ns, Ns//2 = ", Ns, Ns//2)
    idx_end = (Ns-1)//2+1
    # Not including dc mode
    freq_pos = frequencies[1:(Ns-1)//2+1]
    # With dc mode
    freq_dc_pos = frequencies[0:(Ns-1)//2+1]
    
    freq_neg = frequencies[(Ns-1)//2+1:]
    #print("freq_pos = ", freq_pos)
    #print("freq_neg = ", freq_neg)

print("")
abs_yf =  np.abs(yf)
# Scale modes (except dc and Nyquist)
scaled_abs_yf = np.multiply(abs_yf, scale)
single_sided_spec = scaled_abs_yf[0:idx_end]

scaled_yf = np.multiply(yf, scale)
single_sided_spec_complex = scaled_yf[0:idx_end]


#print("np.abs(yf[0:N//2]) = ", np.abs(yf[0:N//2]))

plt.plot(x, y)
plt.show()

#print("")
#print("")
#print("frequencies[0:idx_end] = ", frequencies[0:idx_end])
#print("single_sided_spec = ", single_sided_spec)

plt.plot(frequencies, abs_yf, 'bs')
plt.xlabel('fx')
plt.plot(frequencies[0:idx_end], single_sided_spec, 'ro')
plt.show()

# Plot single-sided spectrum
plt.plot(2.*np.pi*frequencies, abs_yf, 'bs')
plt.xlabel('kx')
plt.plot(2.*np.pi*frequencies[0:idx_end], single_sided_spec, 'ro')
plt.show()

# Reconstruct signal from coefs
#print("")
#print("frequencies[1], frequencies[2], frequencies[3]", frequencies[1], frequencies[2], frequencies[3])
#print("single_sided_spec[0], single_sided_spec[1], single_sided_spec[2] = ", single_sided_spec[0], single_sided_spec[1], single_sided_spec[2])
#print("")
basis1 = scaled_yf[1]*np.exp(1j*2*np.pi*freq_dc_pos[1]*x)
#basis2 = np.exp(1j*2*np.pi*frequencies[2]*x)
#basis3 = np.exp(1j*2*np.pi*frequencies[3]*x)

signal_recon = 0.
for ii in range(0, len(freq_dc_pos)):
    basis = np.exp(1j*2*np.pi*freq_dc_pos[ii]*x)
    signal_recon = signal_recon + single_sided_spec_complex[ii]*basis

signal_recon22 = 0.
for ii in range(0, len(frequencies)):
    basis = np.exp(1j*2*np.pi*frequencies[ii]*x)
    signal_recon22 = signal_recon22 + yf[ii]*basis

    
#signal_recon = basis1

# Take real part of signal
signal_recon = signal_recon.real
signal_recon22 = signal_recon22.real

#print("signal_recon = ", signal_recon.real)
#print("")
#print("original signal: ", y)

print("Max. difference between original and reconstructed signal: ", np.amax(np.abs(signal_recon-y)))
print("Max. difference between original and reconstructed signal 22: ", np.amax(np.abs(signal_recon22-y)))
print("")


import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq, fft2, fftshift
import numpy as np

# Number of sample points in x and y
Ns_x = 21
Ns_y = 61

# wavenumbers in x and y
kx = 2.
ky = 3.

# kx2 = -4.
# ky2 = -6.

# kx3 = 6.
# ky3 = 9.

try: kx2
except NameError: kx2 = None
try: ky2
except NameError: ky2 = None
try: kx3
except NameError: kx3 = None
try: ky3
except NameError: ky3 = None

Lamx = 2.*np.pi/kx
Lamy = 2.*np.pi/ky

print("")
print("Lamx = ", Lamx)
print("Lamy = ", Lamy)

# By analogy with the time frequency, the space frequency is fx = 1./Lamx
fx = 1./Lamx
fy = 1./Lamy
print("")
print("kx, ky = ", kx, ky)
print("Space frequency are fx, fy = ", fx, fy)
print("")

if (kx2 == None):
    pass
else:
    Lamx2 = 2.*np.pi/kx2
    Lamy2 = 2.*np.pi/ky2
    fx2 = 1./Lamx2
    fy2 = 1./Lamy2
    print("fx2, fy2 = ", fx2, fy2)

if (kx3 == None):
    pass
else:
    Lamx3 = 2.*np.pi/kx3
    Lamy3 = 2.*np.pi/ky3
    fx3 = 1./Lamx3
    fy3 = 1./Lamy3
    print("fx3, fy3 = ", fx3, fy3)

# samples
x = np.linspace(0.0, Lamx, Ns_x, endpoint=False)
y = np.linspace(0.0, Lamy, Ns_y, endpoint=False)

X, Y = np.meshgrid(x, y)

dx = x[1]-x[0]
dy = y[1]-y[0]

Fs_x = 1/dx
Fs_y = 1/dy
print("")
print("Sampling frequencies are Fs_x, Fs_y = ", Fs_x, Fs_y)
print("")

fNyquist_x = Fs_x/2.
fNyquist_y = Fs_y/2.

print("fNyquist_x, fNyquist_y = ", fNyquist_x, fNyquist_y)
print("")

# Create signal
z  = np.sin(kx*X + ky*Y)
#z  = np.sin(kx*X + ky*Y) + np.sin(kx2*X + ky2*Y) + np.sin(kx3*X + ky3*Y)

# To check that normalization fft(y)/(Ns_x*Ns_y) is correct, use z = z**3. and check that the average in the spectrum = 0.5
# use for instance kx=2, ky=3
take_cube=False
if (take_cube):
    z = z**3.
    freq3_x = 3.*fx
    freq3_y = 3.*fy
    print("")
    print("Frequency f3x = %21.11e and f3y = %21.11e should also be present" % (freq3_x, freq3_y))
    print("")

print("z.shape = ", z.shape)

#zz = X * 2 + 4 * Y**2
plt.contourf(X, Y, z, cmap = 'jet')
plt.colorbar()
plt.show()

# Compute and normalize fft
zf = fft2(z)/(Ns_x*Ns_y)
# Shift
zf = fftshift(zf)

print("zf.shape[0], Ns_y = ", zf.shape[0], Ns_y)
print("zf.shape[1], Ns_x = ", zf.shape[1], Ns_x)

# Remember with meshgrid: i is row index so typically your j index in a real mesh, while j is your column index (typically i-index in grid) 
freq_x = fftfreq(Ns_x,d=dx)
freq_y = fftfreq(Ns_y,d=dy)

print("")
print("freq_x (before shift) = ", freq_x)
print("")
print("freq_y (before shift) = ", freq_y)
print("")

freq_x = fftshift(freq_x)
freq_y = fftshift(freq_y)

print("")
print("freq_x (after shift) = ", freq_x)
print("")
print("freq_y (after shift) = ", freq_y)
print("")

# np.asscalar is deprecated ==> use .item() instead
#idx_x0 = np.asscalar(np.argwhere(freq_x == 0))
#idx_y0 = np.asscalar(np.argwhere(freq_y == 0))

#print("idx_x0 = ", idx_x0)
#print("idx_y0 = ", idx_y0)

idx_x0 = np.argwhere(freq_x == 0).item(0)
idx_y0 = np.argwhere(freq_y == 0).item(0)

#print("idx_x0 = ", idx_x0)
#print("idx_y0 = ", idx_y0)

print("np.abs(zf[idx_y0, idx_x0]) = ", np.abs(zf[idx_y0, idx_x0]))

#print("type(idx_x0) = ", type(idx_x0))
#print("np.argwhere(freq_x == 0) = ", np.argwhere(freq_x == 0))
#print("np.argwhere(freq_y == 0) = ", np.argwhere(freq_y == 0))

freq_xx, freq_yy = np.meshgrid(freq_x, freq_y)

#random_vals = np.array(data)[random_values.astype(int)]
freq_xx_pos = freq_x[idx_x0:]
freq_yy_pos = freq_y[idx_y0:]

print("")
print("freq_xx_pos = ", freq_xx_pos)
print("")
freq_xx_pos, freq_yy_pos = np.meshgrid(freq_xx_pos, freq_yy_pos)
#idx_x, idx_y = np.meshgrid(np.arange())

abs_zf = np.abs(zf)

#print("zf = ", zf)

plt.contourf(freq_xx, freq_yy, abs_zf, cmap = 'jet')
plt.colorbar()
plt.xlabel('fx')
plt.ylabel('fy')
plt.title('double sided spectrum')
plt.show()

plt.contourf(2.*np.pi*freq_xx, 2.*np.pi*freq_yy, abs_zf, cmap = 'jet')
plt.colorbar()
plt.xlabel('kx')
plt.ylabel('ky')
plt.title('double sided spectrum')
plt.show()

# This still needs some work
plt.contourf(freq_xx_pos, freq_yy_pos, abs_zf[idx_y0:, idx_x0:], cmap = 'jet')
plt.colorbar()
plt.xlabel('fx')
plt.ylabel('fy')
plt.title('single sided spectrum')
plt.show()

#print("zf[idx_x0:, idx_y0:] = ", zf[idx_x0:, idx_y0:])

signal_recon = 0.
for ii in range(0, len(freq_x)):
    for jj in range(0, len(freq_y)):
        #print("ii, jj = ", ii, jj)
        basis = np.exp(1j*2*np.pi*( freq_xx[jj, ii]*X + freq_yy[jj, ii]*Y) )
        signal_recon = signal_recon + zf[jj, ii]*basis

#print("basis.shape = ", basis.shape)

print("")
print("Max. difference between original and reconstructed signal: ", np.amax(np.abs(signal_recon-z)))
print("")






















# from scipy.fft import fft, fftfreq
# import numpy as np
# # Number of sample points
# N = 600
# # sample spacing
# T = 1.0 / 800.0
# x = np.linspace(0.0, N*T, N, endpoint=False)

# x2 = np.linspace(0.0, N*T, N, endpoint=True)

# print("N, len(x), len(x2) = ", N, len(x), len(x2))

# # frequencies are 50 and 80
# y = np.sin(50.0 * 2.0*np.pi*x) + 0.5*np.sin(80.0 * 2.0*np.pi*x)
# yf = fft(y)
# xf = fftfreq(N, T)[:N//2]
# import matplotlib.pyplot as plt

# #print("np.abs(yf[0:N//2]) = ", np.abs(yf[0:N//2]))

# plt.plot(xf, 2.0/N * np.abs(yf[0:N//2]))
# plt.grid()
# plt.show()

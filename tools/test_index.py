
import numpy as np

dp  = np.dtype('d')        # double precision
dpc = np.dtype(np.cdouble) # double precision complex


nz = 50

aa = np.zeros(nz, dp)

for i in range(0, nz):

    print("i = ", i)
    print("aa[i] = ", aa[i])

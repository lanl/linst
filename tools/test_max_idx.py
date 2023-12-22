
import psutil
import sys
import math
import time
import warnings
import numpy as np
dp  = np.dtype('d')        # double precision
dpc = np.dtype(np.cdouble) # double precision complex

print("5%5 = ", 5%5)
print("5%4 = ", 5%4)

a = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]]) + 1j*0.1*np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
aa = np.zeros((3,3,4), dpc)

aa[0] = a
aa[1] = a+10
aa[2] = a+20

print("a = ", a)

print("aa = ",aa)

print("aa[0] =", aa[0])

i = 1
print("aa[:, i, :] = ", aa[:, i, :])

print("Find max for each variable =", np.max( np.abs(aa[:, i, :]), axis=1) )


# Calling psutil.cpu_precent() for 4 seconds
print('The CPU usage is: ', psutil.cpu_percent(0.1))



import psutil
import sys
import math
import time
import warnings
import numpy as np
dp  = np.dtype('d')        # double precision
dpc = np.dtype(np.cdouble) # double precision complex

a = np.array([1, 2, 3])
b = np.array([5, 6, 7])

test = [(a[i]*b[j]) for i in range(3) for j in range(3)]

print("test = ", test)




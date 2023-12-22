
import sys
import math
import time
import warnings
import numpy as np

a = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])

print("a = ", a)
print("a[0,1] = ", a[0,1])

print("a.flatten() = ", a.flatten())

print("np.transpose(a).flatten() = ", np.transpose(a).flatten())


import numpy as np
import random

dp = np.dtype('d')        # double precision

j = np.random.random(1)

print("j = ", j)

k = np.random.random ([300,400])

print("k = ", k)
print("np.amax(k) = ", np.amax(k))


nx = 500
ny = 500

test = np.zeros((nx, ny), dp)

for j in range(0, ny):
    for i in range(0, nx):

        test[i,j] = random.uniform(-1, 1)

average = np.sum(test)/(nx*ny)
print("average = ", average)

test_zero_ave = test - average

print("test = ", test)

print("test_zero_ave = ", test_zero_ave)
print("np.sum(test_zero_ave) = ", np.sum(test_zero_ave))


print("random.uniform(-1, 1) = ", random.uniform(-1, 1))


#syn0 = 2*numpy.random.random((3,1)) - 1

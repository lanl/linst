
import matplotlib
import matplotlib.pyplot as plt

import numpy as np

a = np.arange(6).reshape((3, 2))

print("a = ", a)

b = np.reshape(a, (6, 1))
print("b = ", b)

d = np.reshape(b, (3, 2))
print("d = ", d)

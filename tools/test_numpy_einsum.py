
import numpy as np

u = np.array([1, 2, 3, 4])
gradc = np.array([5, 5, 6, 7])

hello = -np.einsum("i...,i...", u, gradc)

print("np.einsum = ", hello)

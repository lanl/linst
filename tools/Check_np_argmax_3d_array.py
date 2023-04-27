
import numpy as np

array = np.random.rand(3,2)

print("np.argmax(np.abs(array))= ", np.argmax(np.abs(array)))

print("array = ", array)

i,j = np.unravel_index(array.argmax(), array.shape)

print("array[i,j]=", array[i,j])

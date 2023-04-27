
import numpy as np

dp  = np.dtype('d')        # double precision
dpc = np.dtype(np.cdouble) # double precision complex

i4  = np.dtype('i4') # integer 4
i8  = np.dtype('i8') # integer 8

nx = 11
ny = 21
nz = 31

xyz_array = np.zeros((3, nx, ny, nz), dp)
x_array = np.zeros((nx, ny, nz), dp)
y_array = np.zeros((nx, ny, nz), dp)
z_array = np.zeros((nx, ny, nz), dp)

for k in range(0, nz):
    for j in range(0, ny):
        for i in range(0, nx):

            x_array[i,j,k] = 1 + i
            y_array[i,j,k] = 100 + j
            z_array[i,j,k] = 1000 + k

xyz_array[0] = x_array
xyz_array[1] = y_array
xyz_array[2] = z_array

for k in range(0, nz):
    for j in range(0, ny):
        for i in range(0, nx):

            print("xyz_array[0,i,j,k], x_array[i,j,k] = ", xyz_array[0,i,j,k], x_array[i,j,k])
            print("xyz_array[1,i,j,k], y_array[i,j,k] = ", xyz_array[1,i,j,k], y_array[i,j,k])
            print("xyz_array[2,i,j,k], z_array[i,j,k] = ", xyz_array[2,i,j,k], z_array[i,j,k])

            
    


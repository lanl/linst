import os
import numpy as np

dp  = np.dtype('d')  # double precision


print("Hello")

nx = 101
ny = 101

x = np.linspace(0,5,nx)
y = np.linspace(-10,10,ny)

X = np.zeros((nx,ny), dp)
Y = np.zeros((nx,ny), dp)

VXY = np.zeros((nx,ny), dp)

for i in range(0, nx):
    for j in range(0, ny):

        X[i,j] = x[i]
        Y[i,j] = y[j]

        VXY[i,j] = np.sin(X[i,j])*np.cos(Y[i,j])
        
# Write data file out
filename      = "TecFile.dat"
work_dir      = "/home/aph/incompressible-stability-equations/tools"
save_path     = work_dir #+ '/' + folder1

completeName  = os.path.join(save_path, filename)
fileoutFinal  = open(completeName,'w')

zname2D       = 'ZONE T="2-D Zone", I = ' + str(nx) + ', J = ' + str(ny) + '\n'
fileoutFinal.write('TITLE     = "2D DATA"\n')
fileoutFinal.write('VARIABLES = "X" "Y" "VXY"\n')
fileoutFinal.write(zname2D)

for j in range(0, ny):
    for i in range(0, nx):
        fileoutFinal.write("%25.15e %25.15e %25.15e \n" % (X[i,j],Y[i,j],VXY[i,j]) )

fileoutFinal.close()

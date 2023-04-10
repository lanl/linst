import os
import sys
import math
import warnings
import numpy as np
import matplotlib
matplotlib.use('TkAgg') #----> Specify the backend
from scipy import linalg
import matplotlib.pyplot as plt

sys.path.append('../src')
import class_gauss_lobatto as mgl
import class_readinput as rinput
import class_baseflow as mbf
import class_mapping as mma

import pickle

from collections import Counter

# Set automatically font size to 14 and font to "serif"
plt.rcParams['font.size'] = '15'
plt.rc('font', family='serif')

#matplotlib.rcParams['text.usetex'] = True

dp  = np.dtype('d')        # double precision
dpc = np.dtype(np.cdouble) # double precision complex

i4  = np.dtype('i4') # integer 4
i8  = np.dtype('i8') # integer 8

# class Eigenfunction:

#     def __init__(self, ny):
#         self.eigf = np.zeros(5, ny)

def ReadInputFile(file_dir, file_input):

    f_full = os.path.join(file_dir, file_input)
    
    # Instance for class_readinput
    riist   = rinput.ReadInput()
    
    # Read input file
    riist.read_input_file(f_full)

    return riist


def ReadEigenfunction(filename, ny):

    with open(filename, 'r') as fp:
        ny1 = len(fp.readlines())
        
    fp.close()
    
    if ( ny != ny1 ):
        sys.exit("File dimension is not consistent with ny")
    else:
        print("File size consistent with ny")

    ef = np.zeros((5, ny), dpc)
    
    # Loading file data into numpy array and storing it in variable called data_collected
    data_collected = np.loadtxt(filename, skiprows=0, dtype=float)

    y = data_collected[:,0]
    
    # amplitude
    ua = data_collected[:,1]
    va = data_collected[:,2]
    wa = data_collected[:,3]
    pa = data_collected[:,4]
    ra = data_collected[:,5]

    # real parts
    ur = data_collected[:,1+5]
    vr = data_collected[:,2+5]
    wr = data_collected[:,3+5]
    pr = data_collected[:,4+5]
    rr = data_collected[:,5+5]

    # imag parts
    ui = data_collected[:,1+10]
    vi = data_collected[:,2+10]
    wi = data_collected[:,3+10]
    pi = data_collected[:,4+10]
    ri = data_collected[:,5+10]

    u = ur+1j*ui
    v = vr+1j*vi
    w = wr+1j*wi
    p = pr+1j*pi
    r = rr+1j*ri

    ef[0,:] = u
    ef[1,:] = v
    ef[2,:] = w
    ef[3,:] = p
    ef[4,:] = r

    print("")
    print("Max. values of real and imaginary parts of eigenfunction:")
    print("---------------------------------------------------------")
    print("u: max(abs(real(u))), max(abs(imag(u))): ", np.amax(np.abs(u.real)), np.amax(np.abs(u.imag)) )
    print("v: max(abs(real(v))), max(abs(imag(v))): ", np.amax(np.abs(v.real)), np.amax(np.abs(v.imag)) )
    print("w: max(abs(real(w))), max(abs(imag(w))): ", np.amax(np.abs(w.real)), np.amax(np.abs(w.imag)) )
    print("p: max(abs(real(p))), max(abs(imag(p))): ", np.amax(np.abs(p.real)), np.amax(np.abs(p.imag)) )
    print("r: max(abs(real(r))), max(abs(imag(r))): ", np.amax(np.abs(r.real)), np.amax(np.abs(r.imag)) )
    print("")

    return y, ef

    #return y, u, v, w, p, r

def GetDifferentiationMatrix(ny, riist, bsfl_ref, boussinesq):

    # Create instance for class GaussLobatto
    cheb = mgl.GaussLobatto(ny)
    
    # Get Gauss-Lobatto points
    cheb.cheby(ny-1)
    
    # Get spectral differentiation matrices on [1,-1]
    cheb.spectral_diff_matrices(ny-1)
    
    yi = cheb.xc
    
    # Create instance for class Mapping
    map = mma.Mapping(ny)
    
    # Get differentiation matrix
    print("")
    #print("domain extent yinf = ", riist.yinf)
    #print("mapping lmap = ", riist.lmap)
    map.map_shear_layer(riist.yinf, yi, riist.lmap, cheb.DM, bsfl_ref, boussinesq)

    return map


def GenerateGridsXY(nx, ny, nwav, alpha, beta):

    x = np.zeros(nx, dp)
    y = np.zeros(ny, dp)
    
    Lx = 2.0*np.pi/alpha
    Ly = 2.0*np.pi/beta

    x = np.linspace(0.0, nwav*Lx, nx)
    y = np.linspace(0.0, nwav*Ly, ny)

    return x, y

def ComputeDistField_2waves(ny, ef1, ef2, alpha, beta, x, y):

    # We are superposing two waves with (alpha, beta) and (alpha, -beta)

    nx = len(x)
    ny = len(y)
    nz = ef1.shape[1]
    
    nvars = ef1.shape[0]
    
    #print("nz, nvars = ", nz, nvars)

    # This is a real array
    dist_3d = np.zeros((5, nx, ny, nz), dp)

    time = 0.0
    omega_i = 1.0
    # since I set time = 0.0, the value of omega_i is not required ==> set it arbitrary to 1

    for j in range(0, ny):
        for i in range(0, nx):

            for ivar in range(0, nvars): # I write v here too but later overwrites it

                #print("ivar = ", ivar)

                # This is valid for u, w, p, r but not v: because the eigenfunction 1 and 2 are the same for u, w, p, and r
                dist_3d[ivar,i,j,:] = 2.0*np.exp(omega_i*time)*( ef1[ivar,:].real*np.cos(alpha*x[i])*np.cos(beta*y[j]) - ef1[ivar,:].imag*np.sin(alpha*x[i])*np.cos(beta*y[j]))

                # Overwrite v: because eigenfunctions 1 and 2 are different for v
                dist_3d[1,i,j,:] = -2.0*np.exp(omega_i*time)*( ef1[1,:].real*np.sin(alpha*x[i])*np.sin(beta*y[j]) - ef1[1,:].imag*np.cos(alpha*x[i])*np.sin(beta*y[j]))


    return dist_3d
    
def Write3dDistFlow(xg, yg, zg, dist_3d, file_dir):

    nx = len(xg)
    ny = len(yg)
    nz = len(zg)

    max_3d = np.max(dist_3d)
    print("")
    print("max_3d = ", max_3d)

    #print("nx, ny, nz = ", nx, ny, nz)
    
    # Write data file out
    filename      = "DistFlow_3d_tec.dat"
    save_path     = file_dir
    
    completeName  = os.path.join(save_path, filename)
    fileoutFinal  = open(completeName,'w')
    
    zname3D       = 'ZONE T="3-D Zone", I = ' + str(nx) + ', J = ' + str(ny) + ', K = ' + str(nz) + '\n'
    fileoutFinal.write('TITLE     = "3D DATA"\n')
    fileoutFinal.write('VARIABLES = "X" "Y" "Z" "u-dist" "v-dist" "w-dist" "p-dist" "r-dist" \n')
    fileoutFinal.write(zname3D)

    str_o = "%25.15e %25.15e %25.15e %25.15e %25.15e %25.15e %25.15e %25.15e \n"

    for k in range(0, nz):
        for j in range(0, ny):
            for i in range(0, nx):
                #print("dist_3d[:,i,j,k] = ", dist_3d[:,i,j,k])
                #print("x[i], y[j], z[k] = ", xg[i], yg[j], zg[k])
                fileoutFinal.write(str_o % ( xg[i], yg[j], zg[k], dist_3d[0,i,j,k], dist_3d[1,i,j,k], dist_3d[2,i,j,k], dist_3d[3,i,j,k], dist_3d[4,i,j,k] ) )
            
    fileoutFinal.close()
            

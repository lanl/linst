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

from module_utilities import FindResonableYminYmax, compute_inner_prod, \
    get_eigenfunction_derivatives, get_eigenfunction_2nd_derivatives, \
    compute_disturbance_kin_energy, compute_disturbance_dissipation

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

ucmp = 0
vcmp = 1
wcmp = 2
pcmp = 3
rcmp = 4

def ReadInputFile(file_dir, file_input):

    f_full = os.path.join(file_dir, file_input)
    
    # Instance for class_readinput
    riist   = rinput.ReadInput()
    
    # Read input file
    riist.read_input_file(f_full)

    return riist


def ReadEigenfunction(filename, nz):

    with open(filename, 'r') as fp:
        nz1 = len(fp.readlines())
        
    fp.close()
    
    if ( nz != nz1 ):
        sys.exit("File dimension is not consistent with nz")
    else:
        print("File size consistent with nz")

    ef = np.zeros((5, nz), dpc)
    
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

    maxu_real = np.amax(np.abs(u.real))
    maxu_imag = np.amax(np.abs(u.imag))

    maxv_real = np.amax(np.abs(v.real))
    maxv_imag = np.amax(np.abs(v.imag))

    maxw_real = np.amax(np.abs(w.real))
    maxw_imag = np.amax(np.abs(w.imag))

    maxp_real = np.amax(np.abs(p.real))
    maxp_imag = np.amax(np.abs(p.imag))

    maxr_real = np.amax(np.abs(r.real))
    maxr_imag = np.amax(np.abs(r.imag))

    tol_c = 1e-10
    
    if ( maxu_real > tol_c or maxv_real > tol_c ):
        sys.exit("Unexpected real part for u or v")

    if ( maxu_imag < tol_c or maxv_imag < tol_c ):
        sys.exit("Unexpected imag part for u or v")

    if ( maxw_real < tol_c or maxp_real < tol_c or maxr_real < tol_c ):
        sys.exit("Unexpected real part for w, p, or r")

    if ( maxw_imag > tol_c or maxp_imag > tol_c or maxr_imag > tol_c ):
        sys.exit("Unexpected imag part for w, p, or r")

    return y, ef

    #return y, u, v, w, p, r

def GetDifferentiationMatrix(nz, riist, bsfl_ref, boussinesq):

    # Create instance for class GaussLobatto
    cheb = mgl.GaussLobatto(nz)
    
    # Get Gauss-Lobatto points
    cheb.cheby(nz-1)
    
    # Get spectral differentiation matrices on [1,-1]
    cheb.spectral_diff_matrices(nz-1)
    
    yi = cheb.xc
    
    # Create instance for class Mapping
    map = mma.Mapping(nz)
    
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

def ComputeDistField_2waves(ef1, ef2, alpha, beta, x, y, omega_i):

    # We are superposing two waves with (alpha, beta) and (alpha, -beta)

    nx = len(x)
    ny = len(y)
    nz = ef1.shape[1]
    
    nvars = ef1.shape[0]
    
    #print("nz, nvars = ", nz, nvars)

    # This is a real array
    dist_3d = np.zeros((5, nx, ny, nz), dp)

    # since I set time = 0.0, the value of omega_i is not required
    time = 0.0

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
            

def compute_reynolds_stresses_two_wave_case(zgrid, alpha, beta, ef1, omega_i, UseCstVal):

    # for now set time = 0.0
    time = 0.0
    
    pi = np.pi
    nz = ef1.shape[1]
    #print("here nz is: ", nz)

    Rs_uu = np.zeros(nz, dp)
    Rs_vv = np.zeros(nz, dp)
    Rs_ww = np.zeros(nz, dp)
    
    Rs_uv = np.zeros(nz, dp)
    Rs_uw = np.zeros(nz, dp)
    Rs_vw = np.zeros(nz, dp)

    ueig = ef1[ucmp,:]
    veig = ef1[vcmp,:]
    weig = ef1[wcmp,:]
    peig = ef1[pcmp,:]
    reig = ef1[rcmp,:]

    if UseCstVal == 1:
        cst = 4.*pi**2./(alpha*beta)
    else:
        cst = 1.0

    Rs_uw = cst*np.exp(2.*omega_i*time)*compute_inner_prod(ueig, weig)

    Rs_uu = cst*np.exp(2.*omega_i*time)*compute_inner_prod(ueig, ueig)
    Rs_vv = cst*np.exp(2.*omega_i*time)*compute_inner_prod(veig, veig)
    Rs_ww = cst*np.exp(2.*omega_i*time)*compute_inner_prod(weig, weig)

    max_uu = np.amax(np.abs(Rs_uu))
    max_vv = np.amax(np.abs(Rs_vv))
    max_ww = np.amax(np.abs(Rs_ww))

    print("")
    print("Reynolds stresses: maximum values")
    print("================================:")
    print("max(uu) = ", np.amax(np.abs(Rs_uu)))
    print("max(vv) = ", np.amax(np.abs(Rs_vv)))
    print("max(ww) = ", np.amax(np.abs(Rs_ww)))
    print("")
    print("max(uv) = ", np.amax(np.abs(Rs_uv)))
    print("max(uw) = ", np.amax(np.abs(Rs_uw)))
    print("max(vw) = ", np.amax(np.abs(Rs_vw)))

    print("")
    print("max_uu/max_ww = ", max_uu/max_ww)
    print("max_vv/max_ww = ", max_vv/max_ww)
    

    ######################################
    # PLOT REYNOLDS STRESSES
    ######################################

    y = zgrid
    
    ptn = plt.gcf().number + 1

    ymin, ymax = FindResonableYminYmax(Rs_ww, y)
    
    f  = plt.figure(ptn)
    ax = plt.subplot(111)
    # Normal components
    ax.plot(np.real(Rs_uu), y, 'k', linewidth=1.5, label=r"$\overline{u'u'}$")
    ax.plot(np.real(Rs_vv), y, 'r', dashes=[5, 5], linewidth=1.5, label=r"$\overline{v'v'}$")
    ax.plot(np.real(Rs_ww), y, 'b', linewidth=1.5, label=r"$\overline{w'w'}$")
    # Cross components facecolors='none', edgecolors='r'
    ax.plot(np.real(Rs_uv), y, marker='s', markerfacecolor='none', markeredgecolor='g', markevery=5, linestyle = 'None', label=r"$\overline{u'v'}$")
    ax.plot(np.real(Rs_uw), y, 'y', linewidth=1.5, label=r"$\overline{u'w'}$")
    ax.plot(np.real(Rs_vw), y, 'm', dashes=[5, 5], linewidth=1.5, label=r"$\overline{v'w'}$")

    # z-derivatives of Reynolds stresses
    #ax.plot(np.real(d_Rs_uu_dz), y, 'c', linewidth=1.5, label=r"$\dfrac{\partial}{\partial z}\overline{u'u'}$")
    #ax.plot(np.real(d_Rs_ww_dz), y, 'c-.', linewidth=1.5, label=r"$\dfrac{\partial}{\partial z}\overline{w'w'}$")

    plt.xlabel("Reynolds stresses", fontsize=18)
    plt.ylabel("z", fontsize=18)
    plt.gcf().subplots_adjust(left=0.17)
    plt.gcf().subplots_adjust(bottom=0.15)
    plt.ylim([ymin, ymax])
    
    plt.legend(loc="upper right")
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15),   # box with lower left corner at (x, y)
              ncol=3, fancybox=True, shadow=True, fontsize=10)
    
    f.show()

    print("")
    input("Check Two-Wave Reynolds Stresses...")
    

def reynolds_stress_balance_eqs_two_wave_case(zgrid, alpha, beta, ef1, omega_i, map, mtmp, bsfl, bsfl_ref, UseCstVal):

    # I simplified the multiplicative constant 4*pi^2/(alpha*beta) and I have assumes time = 0.0

    # for now set time = 0.0
    time = 0.0

    if UseCstVal == 1:
        cst = 4.*pi**2./(alpha*beta)
    else:
        cst = 1.0


    if   ( mtmp.boussinesq == 1 or mtmp.boussinesq == -3 or mtmp.boussinesq == -4 ):
        print("")
        print("For solvers: mob.boussinesq == 1, -3 or -4")
        Rho   = bsfl.Rho_nd

    ueig = ef1[ucmp,:]
    veig = ef1[vcmp,:]
    weig = ef1[wcmp,:]
    peig = ef1[pcmp,:]
    reig = ef1[rcmp,:]

    D1 = map.D1
    D2 = map.D2

    dudx,dudy,dudz,dvdx,dvdy,dvdz,dwdx,dwdy,dwdz,dpdx,dpdy,dpdz,drdx,drdy,drdz = get_eigenfunction_derivatives(ueig, veig, weig, peig, reig, alpha, beta, D1)
    d2udx2,d2udy2,d2udz2,d2vdx2,d2vdy2,d2vdz2,d2wdx2,d2wdy2,d2wdz2,d2pdx2,d2pdy2,d2pdz2,d2rdx2,d2rdy2,d2rdz2 = get_eigenfunction_2nd_derivatives(ueig, veig, weig, peig, reig, alpha, beta, D2)
    
    Sc = bsfl_ref.Sc
    Re = bsfl_ref.Re
    Fr = bsfl_ref.Fr

    # Reynolds stresses
    Rs_uu = cst*np.exp(2.*omega_i*time)*compute_inner_prod(ueig, ueig)
    Rs_vv = cst*np.exp(2.*omega_i*time)*compute_inner_prod(veig, veig)
    Rs_ww = cst*np.exp(2.*omega_i*time)*compute_inner_prod(weig, weig)
    
    #Rs_uu = compute_inner_prod(ueig, ueig)
    #Rs_vv = compute_inner_prod(veig, veig)
    #Rs_ww = compute_inner_prod(weig, weig)
    
    Rs_uv = 0.0*compute_inner_prod(ueig, veig)
    Rs_uw = 0.0*compute_inner_prod(ueig, weig)
    Rs_vw = 0.0*compute_inner_prod(veig, weig)
    
    ############################
    # Balance Equation for u'u'
    ############################

    # Should not have rho_bsfl for Boussinesq!!!!!
    LHS_uu = 2*omega_i*Rs_uu

    print("max(LHS_uu) = ", np.amax(LHS_uu))
    
    RHS_uu_t1 = -2.*compute_inner_prod(ueig, dpdx)
    
    # Compute Laplacian of u'u'
    Lap_uu = 2.*( compute_inner_prod(dudx, dudx) + compute_inner_prod(dudy, dudy) + compute_inner_prod(dudz, dudz) ) + \
             2.*( compute_inner_prod(ueig, d2udx2) + compute_inner_prod(ueig, d2udy2) + compute_inner_prod(ueig, d2udz2) )
    Lap_uu = Lap_uu/Re

    # Compute dissipation of u'u'
    dissip_uu = 2.*( compute_inner_prod(dudx, dudx) + compute_inner_prod(dudy, dudy) + compute_inner_prod(dudz, dudz) )
    dissip_uu = dissip_uu/Re

    RHS_uu = RHS_uu_t1 + Lap_uu - dissip_uu
        
    print("")
    print("Check energy balancefor u'u' Reynolds stress equation")
    print("-----------------------------------------------------")
    energy_bal_uu = np.amax(np.abs(LHS_uu-RHS_uu))
    print("Energy balance Reynolds stress u'u' = ", energy_bal_uu)
    print("")
    
    ############################
    # Balance Equation for v'v'
    ############################

    # Should not have rho_bsfl for Boussinesq!!!!!
    LHS_vv = 2*omega_i*Rs_vv
    
    RHS_vv_t1 = -2.*compute_inner_prod(veig, dpdy)
    
    # Compute Laplacian of v'v'
    Lap_vv = 2.*( compute_inner_prod(dvdx, dvdx) + compute_inner_prod(dvdy, dvdy) + compute_inner_prod(dvdz, dvdz) ) + \
             2.*( compute_inner_prod(veig, d2vdx2) + compute_inner_prod(veig, d2vdy2) + compute_inner_prod(veig, d2vdz2) )
    Lap_vv = Lap_vv/Re

    # Compute dissipation of v'v'
    dissip_vv = 2.*( compute_inner_prod(dvdx, dvdx) + compute_inner_prod(dvdy, dvdy) + compute_inner_prod(dvdz, dvdz) )
    dissip_vv = dissip_vv/Re

    RHS_vv = RHS_vv_t1 + Lap_vv - dissip_vv

    print("max(LHS_vv), max(RHS_vv_t1), max(Lap_vv), max(dissip_vv), max(RHS_vv)=", \
          np.amax(np.abs(LHS_vv)), np.amax(np.abs(RHS_vv_t1)), np.amax(np.abs(Lap_vv)), np.amax(np.abs(dissip_vv)), np.amax(np.abs(RHS_vv)) )
        
    print("")
    print("Check energy balancefor v'v' Reynolds stress equation")
    print("-----------------------------------------------------")
    energy_bal_vv = np.amax(np.abs(LHS_vv-RHS_vv))
    print("Energy balance Reynolds stress v'v' = ", energy_bal_vv)
    print("")
    
    ############################
    # Balance Equation for w'w'
    ############################

    dpdz_bsfl = -Rho/Fr**2*1
    a3 = np.divide(compute_inner_prod(reig, weig), Rho)
    input("Check dpdz_bsfl and a3 and ww equation")

    # LHS: Should not have rho_bsfl for Boussinesq!!!!!
    LHS_ww = 2*omega_i*Rs_ww

    # RHS
    RHS_ww_t1 = -2.*compute_inner_prod(weig, dpdz)
    RHS_ww_t2 = 2.*np.multiply(a3, dpdz_bsfl)

    # Compute Laplacian of w'w'
    Lap_ww = 2.*( compute_inner_prod(dwdx, dwdx) + compute_inner_prod(dwdy, dwdy) + compute_inner_prod(dwdz, dwdz) ) + \
             2.*( compute_inner_prod(weig, d2wdx2) + compute_inner_prod(weig, d2wdy2) + compute_inner_prod(weig, d2wdz2) )
    Lap_ww = Lap_ww/Re

    # Compute dissipation of w'w'
    dissip_ww = 2.*( compute_inner_prod(dwdx, dwdx) + compute_inner_prod(dwdy, dwdy) + compute_inner_prod(dwdz, dwdz) )
    dissip_ww = dissip_ww/Re

    RHS_ww = RHS_ww_t1 + RHS_ww_t2 + Lap_ww - dissip_ww

    print("")
    print("Check energy balancefor w'w' Reynolds stress equation")
    print("-----------------------------------------------------")
    energy_bal_ww = np.amax(np.abs(LHS_ww-RHS_ww))
    print("Energy balance Reynolds stress w'w' = ", energy_bal_ww)
    print("")

    ######################################
    # MODEL TERMS
    ######################################

    #
    # 1) u'u' equation
    #

    #
    # 2) v'v' equation
    #

    #
    # 3) w'w' equation
    #
    

    ######################################
    # PLOT REYNOLDS STRESSES BALANCE EQ.
    ######################################

    y = zgrid

    ymins = -0.2
    ymaxs = -ymins

    xmins = -40
    xmaxs = 60
    
    #######
    # w'w'
    #######
    
    ptn = plt.gcf().number + 1

    if (ymins == 0.):
        ymins, ymaxs = FindResonableYminYmax(Rs_uu, y)

    max_1 = np.amax(np.abs(RHS_ww_t1))
    
    f  = plt.figure(ptn)
    ax = plt.subplot(111)
    ax.plot(np.real(LHS_ww), y, 'k', linewidth=1.5, label=r"LHS $(\overline{w'w'})$")
    ax.plot(np.real(RHS_ww), y, 'g--', linewidth=1.5, label=r"RHS $(\overline{w'w'})$")

    ax.plot(np.real(RHS_ww_t1), y, 'r', linewidth=1.5, label=r"$-2\overline{w'\dfrac{\partial p'}{\partial z}}$ (RHS)")
    ax.plot(np.real(RHS_ww_t2), y, 'b', linewidth=1.5, label=r"$2a_z \dfrac{\partial \bar{p}}{\partial z}$ (RHS)")

    ax.plot(np.real(Lap_ww), y, 'm', linewidth=1.5, label=r"$\dfrac{1}{Re} \nabla^2 \overline{w'w'}$ (RHS)")
    ax.plot(np.real(dissip_ww), y, 'c', linewidth=1.5, label=r"$\dfrac{2}{Re} \overline{\nabla w' \cdot \nabla w'}$ (RHS)")    

    plt.xlabel("Reynolds stresses balance eq. for w'w'", fontsize=18)
    plt.ylabel("z", fontsize=18)
    plt.gcf().subplots_adjust(left=0.17)
    plt.gcf().subplots_adjust(bottom=0.15)

    plt.xlim([xmins, xmaxs])
    plt.ylim([ymins, ymaxs])
    
    plt.legend(loc="upper right")
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15),   # box with lower left corner at (x, y)
              ncol=3, fancybox=True, shadow=True, fontsize=10)
    
    f.show()


    #######
    # u'u'
    #######
    
    ptn = plt.gcf().number + 1

    #if (ymins == 0.):
    ymins, ymaxs = FindResonableYminYmax(Rs_uu, y)
    
    f  = plt.figure(ptn)
    ax = plt.subplot(111)
    ax.plot(np.real(LHS_uu), y, 'k', linewidth=1.5, label=r"LHS $(\overline{u'u'})$")
    ax.plot(np.real(RHS_uu), y, 'g--', linewidth=1.5, label=r"RHS $(\overline{u'u'})$")

    ax.plot(np.real(RHS_uu_t1), y, 'r', linewidth=1.5, label=r"$-2\overline{u'\dfrac{\partial p'}{\partial x}}$ (RHS)")

    ax.plot(np.real(Lap_uu), y, 'b', linewidth=1.5, label=r"$\dfrac{1}{Re} \nabla^2 \overline{u'u'}$ (RHS)")
    ax.plot(np.real(dissip_uu), y, 'm', linewidth=1.5, label=r"$\dfrac{2}{Re} \overline{\nabla u' \cdot \nabla u'}$ (RHS)")    

    plt.xlabel("Reynolds stresses balance eq. for u'u'", fontsize=18)
    plt.ylabel("z", fontsize=18)
    plt.gcf().subplots_adjust(left=0.17)
    plt.gcf().subplots_adjust(bottom=0.15)

    #plt.xlim([xmins, xmaxs])
    plt.ylim([ymins, ymaxs])
    
    plt.legend(loc="upper right")
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15),   # box with lower left corner at (x, y)
              ncol=3, fancybox=True, shadow=True, fontsize=10)
    
    f.show()

    #######
    # v'v'
    #######
    
    ptn = plt.gcf().number + 1

    #if (ymins == 0.):
    ymins, ymaxs = FindResonableYminYmax(Rs_uu, y)
    
    f  = plt.figure(ptn)
    ax = plt.subplot(111)
    ax.plot(np.real(LHS_vv), y, 'k', linewidth=1.5, label=r"LHS $(\overline{v'v'})$")
    ax.plot(np.real(RHS_vv), y, 'g--', linewidth=1.5, label=r"RHS $(\overline{v'v'})$")

    ax.plot(np.real(RHS_vv_t1), y, 'r', linewidth=1.5, label=r"$-2\overline{v'\dfrac{\partial p'}{\partial y}}$ (RHS)")

    ax.plot(np.real(Lap_vv), y, 'b', linewidth=1.5, label=r"$\dfrac{1}{Re} \nabla^2 \overline{v'v'}$ (RHS)")
    ax.plot(np.real(dissip_vv), y, 'm', linewidth=1.5, label=r"$\dfrac{2}{Re} \overline{\nabla v' \cdot \nabla v'}$ (RHS)")    

    plt.xlabel("Reynolds stresses balance eq. for v'v'", fontsize=18)
    plt.ylabel("z", fontsize=18)
    plt.gcf().subplots_adjust(left=0.17)
    plt.gcf().subplots_adjust(bottom=0.15)

    #plt.xlim([xmins, xmaxs])
    plt.ylim([ymins, ymaxs])
    
    plt.legend(loc="upper right")
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15),   # box with lower left corner at (x, y)
              ncol=3, fancybox=True, shadow=True, fontsize=10)
    
    f.show()

    

    input("Reynolds stress balance equations")

    

def compare_exact_model_terms_a_equation_lst(zgrid, alpha, beta, ef1, omega_i, map, mtmp, bsfl, bsfl_ref):
    
    if   ( mtmp.boussinesq == 1 or mtmp.boussinesq == -3 or mtmp.boussinesq == -4 ):
        Rho  = bsfl.Rho_nd
        Rhop = bsfl.Rhop_nd
        
        Rho2 = np.multiply(Rho, Rho)
        Rho3 = np.multiply(Rho2, Rho)
        
    ueig = ef1[ucmp,:]
    veig = ef1[vcmp,:]
    weig = ef1[wcmp,:]
    peig = ef1[pcmp,:]
    reig = ef1[rcmp,:]

    D1 = map.D1
    D2 = map.D2

    dpdz = np.matmul(D1, peig)
    drdz = np.matmul(D1, reig)

    #Sc = bsfl_ref.Sc
    #Re = bsfl_ref.Re
    Fr = bsfl_ref.Fr

    # I have removed the multiplicative constant 4*pi^2/(alpha*beta) as well as the term exp(2*omega_i*t)

    # Compute v'_d*dp'dz
    vd_dpdz = -np.divide( compute_inner_prod(reig, dpdz), Rho2 )

    # Compute v'_d*p'
    vd_p = -np.divide( compute_inner_prod(reig, peig), Rho2 )

    # Compute d/dz(v'_d*p')
    ddz_vd_p = np.matmul(D1, vd_p)

    # Compute d/dz(v'_d*p') ===> other method
    tbsfl = np.divide(Rhop, Rho3)
    ddz_vd_p_22 = -np.divide( compute_inner_prod(drdz, peig), Rho2 ) + 2.*np.multiply( tbsfl, compute_inner_prod(reig, peig) ) -np.divide( compute_inner_prod(reig, dpdz), Rho2 )

    print("")
    #print("ddz_vd_p = ", ddz_vd_p)
    #print("ddz_vd_p_22 = ", ddz_vd_p_22)
    print("np.amax(np.abs(ddz_vd_p_22-ddz_vd_p)) = ", np.amax(np.abs(ddz_vd_p_22-ddz_vd_p)))
    print("")

    # Compute p'*d/dz(v'_d)
    p_dvddz = -np.divide( compute_inner_prod(peig, drdz), Rho2 ) + 2.*np.multiply( tbsfl, compute_inner_prod(peig, reig) )

    #
    # MODEL TERMS
    #
    bvar = np.divide( compute_inner_prod(reig, reig), Rho2 )

    # Pressure dissipation
    pres_diss = 1./3.*np.multiply(bvar, -Rho)/Fr**2.

    # Pressure diffusion
    dbdz = np.matmul(D1, bvar)

    ke = compute_disturbance_kin_energy(ueig, veig, weig)
    ke_dim = ke*bsfl_ref.Uref**2.

    # Time scale
    t_ref  = bsfl_ref.Lref/bsfl_ref.Uref

    eps_dist = compute_disturbance_dissipation(ueig, veig, weig, peig, reig, zgrid, alpha, beta, D1, D2, bsfl, bsfl_ref, mtmp)[1]
    eps_dim = eps_dist*bsfl_ref.Uref**2./t_ref

    muref  = bsfl_ref.muref
    rhoref = bsfl_ref.rhoref

    visc   = muref/rhoref
    
    l_taylor_dim = np.sqrt(15*visc*np.divide(ke_dim, eps_dim))
    l_taylor_nondim = l_taylor_dim/bsfl_ref.Lref

    l_taylor_nondim2 = np.multiply(l_taylor_nondim, l_taylor_nondim)
    
    pres_diff = -np.matmul(D1, np.multiply(l_taylor_nondim2, dbdz))
    
    ########
    # PLOTS
    ########

    y = zgrid

    ptn = plt.gcf().number + 1

    ymin, ymax = FindResonableYminYmax(l_taylor_dim, y)

    f = plt.figure(ptn)
    ax = plt.subplot(111)
    ax.plot(l_taylor_dim, y, 'k', linewidth=1.5, label=r"Taylor scale $\lambda=\sqrt{15 \nu k/\epsilon}$")

    plt.xlabel("Taylor scale", fontsize=14)
    plt.ylabel('z', fontsize=16)
    plt.gcf().subplots_adjust(left=0.18)
    plt.gcf().subplots_adjust(bottom=0.15)
    plt.ylim([ymin, ymax])

    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15),   # box with lower left corner at (x, y)
              ncol=2, fancybox=True, shadow=True, fontsize=10)
    
    f.show()











    
    
    ptn = plt.gcf().number + 1

    ymin, ymax = FindResonableYminYmax(vd_dpdz, y)
    
    f  = plt.figure(ptn)
    ax = plt.subplot(111)
    ax.plot(np.real(vd_dpdz), y, 'k', linewidth=1.5, label=r"$\overline{ v'\dfrac{\partial p'}{\partial z} }$")
    ax.plot(np.real(ddz_vd_p), y, 'g', linewidth=1.5, label=r"$\dfrac{\partial}{\partial z} \overline{v'p'}$")
    #ax.plot(np.real(ddz_vd_p_22), y, 'r--', linewidth=1.5, label=r"$\dfrac{\partial}{\partial z} \overline{v'_d p'}$ 22222")
    
    ax.plot(np.real(p_dvddz), y, 'b', linewidth=1.5, label=r"$\overline{p' \dfrac{\partial v'}{\partial z}}$")
    ax.plot(np.real(ddz_vd_p-p_dvddz), y, 'r--', linewidth=1.5, label=r"$\dfrac{\partial}{\partial z} \overline{v'p'} - \overline{p' \dfrac{\partial v'}{\partial z}}$")

    #ax.plot(np.real(bvar), y, 'm', linewidth=1.5, label=r"$b=\dfrac{\overline{\rho'\rho'}}{\bar{\rho}^2}$")
    ax.plot(np.real(pres_diss), y, 'c', linewidth=1.5, label=r"$pres_{diss}=\dfrac{1}{3}b \dfrac{\partial \bar{p}}{\partial z}$")
    ax.plot(np.real(pres_diff), y, 'm', linewidth=1.5, label=r"pressure diffusion")

    plt.xlabel(r"a-equation: exact and model terms ($v'=-\rho'/\bar{\rho}^2$)", fontsize=14)
    plt.ylabel("z", fontsize=18)
    plt.gcf().subplots_adjust(left=0.17)
    plt.gcf().subplots_adjust(bottom=0.15)
    plt.ylim([ymin, ymax])
    
    plt.legend(loc="upper right")
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15),   # box with lower left corner at (x, y)
              ncol=3, fancybox=True, shadow=True, fontsize=10)
    
    f.show()

    input("a-equation exact and model terms")





def compute_taylor_and_integral_scales(ke_dim, eps_dim, ke, eps, y, iarr, bsfl_ref):

    # These are dimensional length scales

    

    muref  = bsfl_ref.muref
    rhoref = bsfl_ref.rhoref

    visc   = muref/rhoref

    eps_m = 0.0
    
    # Compute Taylor scale
    iarr.l_taylor_dim = np.sqrt(15*visc*np.divide(ke_dim+eps_m, eps_dim+eps_m))

    # Integral length scale: https://en.wikipedia.org/wiki/Taylor_microscale, + see Daniel model.pdf
    # This is called turbulence length scale S by Schwartzkopf
    iarr.l_integr_dim = np.divide(ke_dim**(1.5)+eps_m, eps_dim+eps_m )

    #
    # Compute integral length scale by dimensionalizing non-dim ke and eps
    #
    
    # Time scale
    t_ref  = bsfl_ref.Lref/bsfl_ref.Uref

    ke_dim_check = ke*bsfl_ref.Uref**2.
    eps_dim_check = eps*bsfl_ref.Uref**2./t_ref

    l_integr_dim_check = np.divide(ke_dim_check**(1.5), eps_dim_check)


    
    # ------------------------------------------------
    # Plot taylor microscale and integral length scale
    # ------------------------------------------------
    
    # Get ymin-ymax
    #ymin, ymax = FindResonableYminYmax(iarr.l_taylor_dim, y)

    ymin = -0.04
    ymax = -ymin

    ptn = plt.gcf().number + 1

    f = plt.figure(ptn)
    ax = plt.subplot(111)
    ax.plot(iarr.l_taylor_dim, y, 'k', linewidth=1.5, label=r"Taylor scale $\lambda=\sqrt{15 \nu k/\epsilon}$")

    plt.xlabel("Taylor scale", fontsize=14)
    plt.ylabel('z', fontsize=16)
    plt.gcf().subplots_adjust(left=0.18)
    plt.gcf().subplots_adjust(bottom=0.15)
    plt.ylim([ymin, ymax])

    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15),   # box with lower left corner at (x, y)
              ncol=2, fancybox=True, shadow=True, fontsize=10)
    
    f.show()

    #
    # Integral length scale
    #
    
    # Get ymin-ymax
    #ymin, ymax = FindResonableYminYmax(iarr.l_integr_dim, y)

    ptn = plt.gcf().number + 1

    f = plt.figure(ptn)
    ax = plt.subplot(111)
    ax.plot(iarr.l_integr_dim, y, 'b', linewidth=1.5, label=r"Integral length scale $L=k^{3/2}/\epsilon$")
    #ax.plot(l_integr_dim_check, y, 'm-.', linewidth=1.5, label=r"Integral length scale (CEHCK!!!) $L=k^{3/2}/\epsilon$")

    plt.xlabel("Integral length scale", fontsize=14)
    plt.ylabel('z', fontsize=16)
    plt.gcf().subplots_adjust(left=0.18)
    plt.gcf().subplots_adjust(bottom=0.15)
    plt.ylim([ymin, ymax])

    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15),   # box with lower left corner at (x, y)
              ncol=2, fancybox=True, shadow=True, fontsize=10)
    
    f.show()

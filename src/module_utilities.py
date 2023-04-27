import os
import sys
import math
import warnings
import numpy as np
import matplotlib
matplotlib.use('TkAgg') #----> Specify the backend
from scipy import linalg
import matplotlib.pyplot as plt

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

def get_mid_idx(ny):

    if (ny % 2) != 0:
        mid_idx = int( (ny-1)/2 )
    else:
        warnings.warn("When ny is even, there is no mid-index")
        input("Continue?")
        print("")
        mid_idx = int ( ny/2 )

    return mid_idx

def get_idx_of_max(eigvals):
    """
    xxxxx
    """
    idx = -666
    max_tmp = 0.
    
    for i in range(0, len(eigvals)):
        if ( eigvals[i].imag > max_tmp ):
            max_tmp = eigvals[i].imag
            idx = i

    if (idx==-666):
        warnings.warn("idx could not be determined in get_idx_of_max!!!!!!!")
        idx = 1
        
    return idx


def get_idx_of_closest_eigenvalue(eigvals, abs_target, target):
    """
    This function finds the index in the eigenvalues array closest to a target eigenvalue
    """

    found = False
    
    indx_imag = -666
    indx_real = -777
    indx_abs  = -888

    print("")

    diff_array_imag = np.abs( eigvals.imag - target.imag )
    diff_array_real = np.abs( eigvals.real - target.real )
    diff_array_abs  = np.abs( eigvals - target )

    sort_index_imag = np.argsort(diff_array_imag)
    sort_index_real = np.argsort(diff_array_real)
    sort_index_abs  = np.argsort(diff_array_abs)

    idx_i = sort_index_imag[0]
    idx_r = sort_index_real[0]
    idx_a = sort_index_abs[0]

    # print("np.sort(diff_array_imag) = ", np.sort(diff_array_imag))
    # print("np.sort(diff_array_real) = ", np.sort(diff_array_real))
    # print("np.sort(diff_array_abs) = ", np.sort(diff_array_abs))

    #print("idx_i, idx_r, idx_a = ", idx_i, idx_r, idx_a)

    # input("1111111")


    List = [idx_i, idx_r, idx_a]

    if ( idx_i != idx_r and idx_i != idx_a and idx_r != idx_a ):
        warnings.warn("Ambiguous situation ==> no eigenvalue close to target!!!!!")
        print("No reliable value found for idx_found ==> setting idx_found to 1")
        print("idx_i, idx_r, idx_a = ", idx_i, idx_r, idx_a)
        idx_found = 1
    else:
        found = True
        idx_found = most_frequent(List)

    #print("type(idx_found) = ", type(idx_found))
        
    #print("most_frequent(List) = ", most_frequent(List))
    #print("idx_i, idx_r, idx_a = ",idx_i, idx_r, idx_a)
    
    # abs_eigs   = np.abs(eigvals)
    # diff_array = np.abs(abs_eigs-abs_target) # calculate the difference array
    
    # sort_index = np.argsort(diff_array)

    # tol = 1.0e-10
    
    # for i in range(0, 5):
    #     val = np.abs(target.imag-eigvals[sort_index[i]].imag)
    #     #print("val, tol: ", val, tol)
    #     if val < tol:
    #         found = True
    #         idx_found = sort_index[i]
    #         break
    #     else:
    #         if i == 4:
    #             print("No reliable value found for idx_found ==> setting idx_found to 1")
    #             idx_found = 1

    if found == True:
        print("Closest eigenvalue to target:", eigvals[idx_found])
        #print("")
    else:
        print("Could not find target")
        print("")

    return idx_found, found

def most_frequent(List):
    occurence_count = Counter(List)
    return occurence_count.most_common(1)[0][0]
    
def write_out_eigenvalues(eigvals, ny):

    count = 0
    for i in range(len(eigvals)):
        if np.abs(eigvals[i]) < 1e7:
            count = count+1

    eigvals_ = np.zeros(count, dpc)

    count = 0
    for i in range(len(eigvals)):
        if np.abs(eigvals[i]) < 1e7:
            eigvals_[count] = eigvals[i]
            count = count + 1

    eigvals_real = eigvals_.real
    eigvals_imag = eigvals_.imag

    data_out = np.column_stack([eigvals_real, eigvals_imag])
    datafile_path = "./eigenvalues_ny" + str(ny) + ".txt" 
    np.savetxt(datafile_path , data_out, fmt=['%21.11e','%21.11e'])

def plot_baseflow(ny, y, yi, U, Up, D1):

    ptn = plt.gcf().number
    
    #plt.rc('text', usetex=True)
    #plt.rcParams["mathtext.fontset"]
    
    # f = plt.figure(ptn)
    # plt.plot(np.zeros(ny), y, 'bs', markerfacecolor='none', label="Grid pts")
    # plt.plot(np.zeros(ny), yi, 'r+', label=" Chebyshev pts")
    # plt.xlabel('')
    # plt.ylabel('y/yi')
    # plt.title('Points distributions')
    # plt.legend(loc="upper right")
    # f.show()

    # ptn=ptn+1

    delta = 1
    
    f = plt.figure(ptn)
    plt.plot(U, y*delta, 'ks', markerfacecolor='none', label="U [-]", linewidth=1.5)
    plt.xlabel(r'$\bar{U}$', fontsize=20)
    #plt.xlim([-0.1, 1.1])
    #plt.ylim([-10, 10])
    plt.ylabel('y', fontsize=20)
    plt.gcf().subplots_adjust(left=0.17)
    plt.gcf().subplots_adjust(bottom=0.15)
    #plt.title('Baseflow U-velocity profile')
    #plt.legend(loc="upper right")
    f.show()

    ptn=ptn+1

    Up_num = np.matmul(D1, U)

    f = plt.figure(ptn)
    plt.plot(Up, y, 'k', markerfacecolor='none', label=r"$\frac{d\bar{U}}{dy}$ (Analytic)", linewidth=1.5)
    plt.plot(Up_num, y, 'r--', markerfacecolor='none', label=r"$\frac{d\bar{U}}{dy}$ (Numerical)", linewidth=1.5)
    plt.xlabel(r"$d\bar{U}/dy$", fontsize=20)
    #plt.xlim([-0.1, 0.5])
    #plt.ylim([-10, 10])    
    plt.ylabel('y', fontsize=20)
    plt.gcf().subplots_adjust(left=0.17)
    plt.gcf().subplots_adjust(bottom=0.15)
    #plt.title("Baseflow U' profile")
    plt.legend(loc="upper right")
    f.show()

    input("Checking baseflow")

def plot_eigvals(eigvals):

    ptn = plt.gcf().number + 1
    
    f = plt.figure(ptn)
    plt.plot(eigvals.real, eigvals.imag, 'ks', markerfacecolor='none')
    plt.xlabel(r'$\omega_r$', fontsize=20)
    plt.ylabel(r'$\omega_i$', fontsize=20)
    plt.gcf().subplots_adjust(left=0.18)
    plt.gcf().subplots_adjust(bottom=0.13)
    #plt.title('Eigenvalue spectrum')
    plt.xlim([-10, 10])
    #plt.ylim([-1, 1])
    f.show()

def plot_phase(phase, str_var, y):

    ptn = plt.gcf().number + 1
    
    f1 = plt.figure(ptn)
    plt.plot(phase, y, 'bs-')
    plt.xlabel('phase(' + str_var + ')')
    plt.ylabel('y')
    plt.gcf().subplots_adjust(left=0.16)
    plt.gcf().subplots_adjust(bottom=0.13)

    f1.show()

def plot_real_imag_part(eig_fct, str_var, y, rt_flag, mob, mv):

    ptn = plt.gcf().number + 1

    # Get ymin-ymax
    ymin, ymax = FindResonableYminYmax(eig_fct, y)

    symbols = 0

    f1 = plt.figure(ptn)
    if (rt_flag and mob.boussinesq == -555):
        plt.plot(y, eig_fct.real, 'k-', linewidth=1.5)
        if symbols==1:
            plt.plot(y, eig_fct.real, 'ks-', linewidth=1.5)
        plt.xlabel('x', fontsize=16)
        if str_var == "u":
            plt.ylabel(r'$\hat{u}_r$', fontsize=16)
        elif str_var == "v":
            plt.ylabel(r'$\hat{v}_r$', fontsize=16)
        elif str_var == "w":
            plt.ylabel(r'$\hat{w}_r$', fontsize=16)
        elif str_var == "p":
            plt.ylabel(r'$\hat{p}_r$', fontsize=16)
        elif str_var == "r":
            plt.ylabel(r'$\hat{\rho}_r$', fontsize=16)
        else:
            plt.ylabel("real(" +str_var + ")", fontsize=16)
            print("Not an expected string!!!!!")
    else:
        plt.plot(eig_fct.real, y, 'k-', linewidth=1.5)
        if symbols==1:
            plt.plot(eig_fct.real, y, 'ks-', linewidth=1.5)
        plt.ylabel('y', fontsize=16)
        if str_var == "u":
            plt.xlabel(r'$\hat{u}_r$', fontsize=16)
        elif str_var == "v":
            plt.xlabel(r'$\hat{v}_r$', fontsize=16)
        elif str_var == "w":
            plt.xlabel(r'$\hat{w}_r$', fontsize=16)
        elif str_var == "p":
            plt.xlabel(r'$\hat{p}_r$', fontsize=16)
        elif str_var == "r":
            plt.xlabel(r'$\hat{\rho}_r$', fontsize=16)
        else:
            plt.xlabel("real(" +str_var + ")", fontsize=16)
            print("Not an expected string!!!!!")

    #if (mv==0.):
    #    plt.ylim([ymin, ymax])
    #else:
    #    plt.ylim([-mv, mv])
    
    f1.show()
    f1.subplots_adjust(bottom=0.16)
    f1.subplots_adjust(left=0.19)

    #plt.gcf().subplots_adjust(left=0.19)
    #plt.gcf().subplots_adjust(bottom=0.16)


    f2 = plt.figure(ptn+1)

    if (rt_flag and mob.boussinesq == -555):
        plt.plot(y, eig_fct.imag, 'k-', linewidth=1.5)
        if symbols==1:
            plt.plot(y, eig_fct.imag, 'ks-', linewidth=1.5)
        plt.xlabel('x', fontsize=16)
        if str_var == "u":
            plt.ylabel(r'$\hat{u}_i$', fontsize=16)
        elif str_var == "v":
            plt.ylabel(r'$\hat{v}_i$', fontsize=16)
        elif str_var == "w":
            plt.ylabel(r'$\hat{w}_i$', fontsize=16)
        elif str_var == "p":
            plt.ylabel(r'$\hat{p}_i$', fontsize=16)
        elif str_var == "r":
            plt.ylabel(r'$\hat{\rho}_i$', fontsize=16)
        else:
            plt.ylabel("imag(" +str_var + ")", fontsize=16)
            print("Not an expected string!!!!!")
    else:
        if symbols==1:
            plt.plot(eig_fct.imag, y, 'ks-', linewidth=1.5)
        plt.plot(eig_fct.imag, y, 'k-', linewidth=1.5)
        plt.ylabel('y', fontsize=16)
        if str_var == "u":
            plt.xlabel(r'$\hat{u}_i$', fontsize=16)
        elif str_var == "v":
            plt.xlabel(r'$\hat{v}_i$', fontsize=16)
        elif str_var == "w":
            plt.xlabel(r'$\hat{w}_i$', fontsize=16)
        elif str_var == "p":
            plt.xlabel(r'$\hat{p}_i$', fontsize=16)
        elif str_var == "r":
            plt.xlabel(r'$\hat{\rho}_i$', fontsize=16)
        else:
            plt.xlabel("imag(" +str_var + ")", fontsize=16)
            print("Not an expected string!!!!!")


    #if (mv==0.):
    #    plt.ylim([ymin, ymax])
    #else:
    #    plt.ylim([-mv, mv])

    f2.subplots_adjust(bottom=0.16)
    f2.subplots_adjust(left=0.19)
    f2.show()


def plot_amplitude(eig_fct, str_var, y):

    ptn = plt.gcf().number + 1

    mv = 20

    f1 = plt.figure(ptn)
    plt.plot(np.abs(eig_fct,), y, 'k', linewidth=1.5)
    if str_var == "u":
        plt.xlabel(r'$|\hat{u}|$', fontsize=20)
    elif str_var == "v":
        plt.xlabel(r'$|\hat{v}|$', fontsize=20)
    elif str_var == "w":
        plt.xlabel(r'$|\hat{w}|$', fontsize=20)
    elif str_var == "p":
        plt.xlabel(r'$|\hat{p}|$', fontsize=20)
    else:
        plt.xlabel("|" +str_var + "|", fontsize=20)
        print("Not an expected string!!!!!")
    plt.ylabel('y', fontsize=20)
    plt.gcf().subplots_adjust(left=0.16)
    plt.gcf().subplots_adjust(bottom=0.16)
    #plt.xlim([-1, 1])
    #plt.ylim([-mv, mv])
    f1.show()


def plot_two_vars_amplitude(eig_fct1, eig_fct2, str_var1, str_var2, y):

    ptn = plt.gcf().number + 1

    mv = 20

    f1 = plt.figure(ptn)
    # First variable
    plt.plot(np.abs(eig_fct1), y, 'k', linewidth=1.5)
    if str_var1 == "u":
        plt.xlabel(r'$|\hat{u}|$', fontsize=20)
    elif str_var1 == "v":
        plt.xlabel(r'$|\hat{v}|$', fontsize=20)
    elif str_var1 == "w":
        plt.xlabel(r'$|\hat{w}|$', fontsize=20)
    elif str_var1 == "p":
        plt.xlabel(r'$|\hat{p}|$', fontsize=20)
    else:
        plt.xlabel("|" +str_var + "|", fontsize=20)
        print("Not an expected string!!!!!")
    # Second variable
    plt.plot(np.abs(eig_fct2), y, 'b--', linewidth=1.5)
    if str_var2 == "u":
        plt.xlabel(r'$|\hat{u}|$', fontsize=20)
    elif str_var2 == "v":
        plt.xlabel(r'$|\hat{v}|$', fontsize=20)
    elif str_var2 == "w":
        plt.xlabel(r'$|\hat{w}|$', fontsize=20)
    elif str_var2 == "p":
        plt.xlabel(r'$|\hat{p}|$', fontsize=20)
    else:
        plt.xlabel("|" +str_var + "|", fontsize=20)
        print("Not an expected string!!!!!")

    plt.ylabel('y', fontsize=20)
    plt.gcf().subplots_adjust(left=0.16)
    plt.gcf().subplots_adjust(bottom=0.16)
    #plt.xlim([-1, 1])
    #plt.ylim([-mv, mv])
    f1.show()


def plot_four_vars_amplitude(eig_fct1, eig_fct2, eig_fct3, eig_fct4, str_var1, str_var2, str_var3, str_var4, y):

    ptn = plt.gcf().number + 1

    mv = 20

    f1 = plt.figure(ptn)
    ax = f1.add_subplot(111)
    # First variable
    plt.plot(np.abs(eig_fct1), y, 'k-', linewidth=1.5, label="|u|")

    # Second variable
    plt.plot(np.abs(eig_fct2), y, 'k--', linewidth=1.5, label="|v|")

    # Third variable
    plt.plot(np.abs(eig_fct3), y, 'k:', linewidth=1.5, label="|w|")

    # Fourth variable
    plt.plot(np.abs(eig_fct4), y, 'k-.', linewidth=1.5, label="|p|")

    plt.ylabel('y', fontsize=20)
    plt.gcf().subplots_adjust(left=0.16)
    plt.gcf().subplots_adjust(bottom=0.16)
    #plt.xlim([-1, 1])
    #plt.ylim([-mv, mv])
    f1.show()

    plt.legend(loc="center right")

    # ratio = 1.0
    # xleft, xright = ax.get_xlim()
    # ybottom, ytop = ax.get_ylim()
    # ax.set_aspect(abs((xright-xleft)/(ybottom-ytop))*ratio)

    # plt.savefig("Plane_Poiseuille_Current",bbox_inches='tight')

def plot_five_vars_amplitude(eig_fct1, eig_fct2, eig_fct3, eig_fct4, eig_fct5, str_var1, str_var2, str_var3, str_var4, str_var5, y):

    zmin = -10#-0.2
    zmax = -zmin
    
    ptn = plt.gcf().number + 1

    mv = 20

    f1 = plt.figure(ptn)
    ax = f1.add_subplot(111)

    fac_u = 1/np.max(np.abs(np.abs(eig_fct1)))

    fac_w = 1/np.max(np.abs(np.abs(eig_fct3)))
    fac_p = 1/np.max(np.abs(np.abs(eig_fct4)))

    fac_r = 1/np.max(np.abs(np.abs(eig_fct5)))

    print("fac_u = ", fac_u)
    print("fac_w = ", fac_w)
    print("fac_p = ", fac_p)
    print("fac_r = ", fac_r)
    
    # First variable
    labelu = "|u| ( scaling = %s )" % str('{:.2e}'.format(fac_u)) 
    plt.plot(np.abs(eig_fct1)*fac_u, y, 'k-', linewidth=1.5, label=labelu)

    # Second variable
    #labelv = "|v| ( scaling = %s )" % str('{:.2e}'.format(fac_v)) 
    #plt.plot(np.real(eig_fct2)*fac_v, y, 'k--', linewidth=1.5, label=labelv)

    # Third variable
    labelw = "|w| ( scaling = %s )" % str('{:.2e}'.format(fac_w)) 
    plt.plot(np.abs(eig_fct3)*fac_w, y, 'b-', linewidth=1.5, label=labelw)
    #plt.plot(np.abs(eig_fct3)*fac_w, y, 'k:', linewidth=1.5, label=labelw)

    # Fourth variable
    labelp = "|p| ( scaling = %s )" % str('{:.2e}'.format(fac_p)) 
    plt.plot(np.abs(eig_fct4)*fac_p, y, 'm-', linewidth=1.5, label=labelp)
    #plt.plot(np.abs(eig_fct4)*fac_p, y, 'k-.', linewidth=1.5, label=labelp)

    # Fifth variable
    plt.plot(np.abs(eig_fct5)*fac_r, y, 'r-', linewidth=1.5, label="|r|")

    plt.ylabel('y', fontsize=20)
    plt.gcf().subplots_adjust(left=0.16)
    plt.gcf().subplots_adjust(bottom=0.16)
    plt.xlim([-1, 1])
    #plt.ylim([zmin, zmax])
    f1.show()

    plt.legend(loc="upper right")

    #ratio = 1.0
    #xleft, xright = ax.get_xlim()
    #ybottom, ytop = ax.get_ylim()
    #ax.set_aspect(abs((xright-xleft)/(ybottom-ytop))*ratio)

    #plt.savefig("Plane_Poiseuille_Current",bbox_inches='tight')

def plot_imag_omega_vs_alpha(omega, str_var, alpha, alp_m, ome_m):

    ptn = plt.gcf().number + 1
    
    max_om = 1.02*np.max(omega.imag)
    
    f1 = plt.figure(ptn)
    plt.plot(alpha, omega.imag, 'b', label="Current")
    plt.plot(alp_m, ome_m, 'rs', label="Michalke (1964)")
    plt.xlabel(r'$\alpha$', fontsize=20)
    plt.ylabel(r'$\omega_i$', fontsize=20)
    plt.xlim([-0.02, 1.02])
    plt.ylim([-0.002, max_om])
    plt.legend(loc="lower center")
    plt.gcf().subplots_adjust(left=0.16)
    plt.gcf().subplots_adjust(bottom=0.13)
    f1.show()

    input("debug all omega")


def get_normalize_eigvcts(ny, target1, idx_tar1, alpha, map, mob, bsfl, bsfl_ref, plot_eigvcts, rt_flag, q_eigvect, Local):
    
    #y  = map.y
    #D1 = map.D1

    if (Local):
        ueig_vec    = q_eigvect[0*ny:1*ny]
        veig_vec    = q_eigvect[1*ny:2*ny]
        weig_vec    = q_eigvect[2*ny:3*ny]
        peig_vec    = q_eigvect[3*ny:4*ny]

        if (rt_flag):
            if ( mob.boussinesq == -555 ):
                ueig_vec    = q_eigvect[0*ny:1*ny]
                veig_vec    = q_eigvect[1*ny:2*ny]
                reig_vec    = q_eigvect[2*ny:3*ny]
                peig_vec    = q_eigvect[3*ny:4*ny]
                
                weig_vec    = 0.0*ueig_vec
            else:
                reig_vec    = q_eigvect[4*ny:5*ny]
        else:
            reig_vec    = 0.0*ueig_vec
            
    else:
        ueig_vec    = q_eigvect[0*ny:1*ny]
        veig_vec    = q_eigvect[1*ny:2*ny]
        weig_vec    = q_eigvect[2*ny:3*ny]
        peig_vec    = q_eigvect[3*ny:4*ny]

        if (rt_flag):
            if ( mob.boussinesq == -555 ):
                ueig_vec    = q_eigvect[0*ny:1*ny]
                veig_vec    = q_eigvect[1*ny:2*ny]
                reig_vec    = q_eigvect[2*ny:3*ny]
                peig_vec    = q_eigvect[3*ny:4*ny]
                
                weig_vec    = 0.0*ueig_vec
            else:
                reig_vec    = q_eigvect[4*ny:5*ny]
        else:
            reig_vec    = 0.0*ueig_vec
    
    # Normalize eigenvectors
    norm_s, ueig_vec, veig_vec, weig_vec, peig_vec, reig_vec = normalize_eigenvectors(ueig_vec, veig_vec, weig_vec, peig_vec, reig_vec, rt_flag)

    return norm_s, ueig_vec, veig_vec, weig_vec, peig_vec, reig_vec

    
def unwrap_shift_phase(ny, ueig, veig, weig, peig, reig, Shift, map, mob, rt_flag):
    
    phase_u = np.zeros(ny, dp)
    phase_v = np.zeros(ny, dp)
    phase_w = np.zeros(ny, dp)
    phase_p = np.zeros(ny, dp)
    phase_r = np.zeros(ny, dp)
    
    phase_u = np.arctan2(ueig.imag, ueig.real)
    phase_v = np.arctan2(veig.imag, veig.real)
    phase_w = np.arctan2(weig.imag, weig.real)
    phase_p = np.arctan2(peig.imag, peig.real)
    phase_r = np.arctan2(reig.imag, reig.real)
    
    amp_u   = np.abs(ueig)
    amp_v   = np.abs(veig)
    amp_w   = np.abs(weig)
    amp_p   = np.abs(peig)
    amp_r   = np.abs(reig)
    
    phase_u_uwrap = np.unwrap(phase_u)
    phase_v_uwrap = np.unwrap(phase_v)
    phase_w_uwrap = np.unwrap(phase_w)
    phase_p_uwrap = np.unwrap(phase_p)
    phase_r_uwrap = np.unwrap(phase_r)
    
    # When I take phase_ref as phase_v_uwrap ==> I get u and v symmetric/anti-symmetric
    # When I take phase_ref as phase_u_uwrap ==> v is not symmetric/anti-symmetric
    print("Reference phase taken from v-velocity")
    print("")

    mid_idx       = get_mid_idx(ny)
    phase_ref     = phase_v_uwrap[mid_idx]
    
    phase_u_uwrap = phase_u_uwrap - phase_ref
    phase_v_uwrap = phase_v_uwrap - phase_ref
    phase_w_uwrap = phase_w_uwrap - phase_ref
    phase_p_uwrap = phase_p_uwrap - phase_ref
    phase_r_uwrap = phase_r_uwrap - phase_ref
    
    # Plot some results
    Plot_this = False
    if (Plot_this):    
        plot_phase(phase_u, "phase_u", map.y)
        plot_phase(phase_v, "phase_v", map.y)
    
    #print("exp(1j*phase) = ", np.exp(1j*phase_u))
    #print("exp(1j*phase_unwrapped) = ", np.exp(1j*phase_u_uwrap))
    
    #print("np.max( np.abs( np.exp(1j*phase_u) - np.exp(1j*phase_u_uwrap ))) = ", np.max(np.abs( np.exp(1j*phase_u) - np.exp(1j*phase_u_uwrap ))))
    #print("np.max( np.abs( np.exp(1j*phase_u)) - np.abs( np.exp(1j*phase_u_uwrap )) ) = ", np.max( np.abs( np.exp(1j*phase_u) ) - np.abs( np.exp(1j*phase_u_uwrap ) ) ) )
        
    if Shift == 1:
        ueig_ps = amp_u*np.exp(1j*phase_u_uwrap)
        veig_ps = amp_v*np.exp(1j*phase_v_uwrap)
        weig_ps = amp_w*np.exp(1j*phase_w_uwrap)
        peig_ps = amp_p*np.exp(1j*phase_p_uwrap)
        reig_ps = amp_r*np.exp(1j*phase_r_uwrap)
    else:
        ueig_ps = ueig
        veig_ps = veig
        weig_ps = weig
        peig_ps = peig
        reig_ps = reig
        
    WriteThisNow = 0
    if (WriteThisNow==1):
        print("")
        print("np.max(np.real(ueig_ps)) = ", np.max(np.real(ueig_ps)))
        print("np.max(np.abs(ueig_ps)) = ", np.max(np.abs(ueig_ps)))
        print("")
        print("")
        print("np.max(np.real(weig_ps)) = ", np.max(np.real(weig_ps)))
        print("np.max(np.abs(weig_ps)) = ", np.max(np.abs(weig_ps)))
        print("")
        print("np.max(np.real(peig_ps)) = ", np.max(np.real(peig_ps)))
        print("np.max(np.abs(peig_ps)) = ", np.max(np.abs(peig_ps)))
        print("")
        print("np.max(np.real(reig_ps)) = ", np.max(np.real(reig_ps)))
        print("np.max(np.abs(reig_ps)) = ", np.max(np.abs(reig_ps)))
        print("")
        
    amp_u_ps = np.abs(ueig_ps)
    amp_v_ps = np.abs(veig_ps)
    amp_w_ps = np.abs(weig_ps)
    amp_p_ps = np.abs(peig_ps)
    amp_r_ps = np.abs(reig_ps)
    
    #ueig_from_continuity = -np.matmul(map.D1, veig_ps)/(1j*alpha)

    if (Plot_this):
        plot_real_imag_part(ueig_ps, "u", map.y, rt_flag, mob)
        #plot_real_imag_part(veig_ps, "v", map.y, rt_flag, mob)
        plot_real_imag_part(weig_ps, "w", map.y, rt_flag, mob)
        plot_real_imag_part(peig_ps, "p", map.y, rt_flag, mob)
        plot_real_imag_part(reig_ps, "r", map.y, rt_flag, mob)
    
        plot_five_vars_amplitude(ueig_ps, veig_ps, weig_ps, peig_ps, reig_ps, "u", "v", "w", "p", "r", map.y)

    return ueig_ps, veig_ps, weig_ps, peig_ps, reig_ps
    
    
def plot_eigvcts(ny, eigvects, target1, idx_tar1, alpha, map, mob, bsfl, bsfl_ref, plot_eigvcts, rt_flag, q_eigvect, Local):

    # Compute phase for v-eigenfunction
    #mid_idx = get_mid_idx(ny)
    
    #phase_vvel = np.arctan2( veig_vec.imag, veig_vec.real)    
    #phase_vvel = phase_vvel - phase_vvel[mid_idx]

    #veig_vec_polar = np.abs(veig_vec)*np.exp(1j*phase_vvel)

    #print("np.max(np.abs(veig_vec-veig_vec_polar)) = ", np.max(np.abs(veig_vec-veig_vec_polar)))

    #dpeig_vec_dx = 1j*alpha*peig_vec 
    #dpeig_vec_dy = np.matmul(D1, peig_vec)


    #updppdx = ueig_vec + dpeig_vec_dx
    #vpdppdy = veig_vec + dpeig_vec_dy

    DoThis = False

    if (DoThis):
        Phi_vec      = -veig_vec/(1j*alpha)
        Phi_vec_new  = -veig_vec_polar/(1j*alpha)
        
        Phi_vec_new  = Phi_vec_new/1j # ADDED otherwise my real part was Michalke imaginary
        
        veig_vec_new = -Phi_vec_new*1j*alpha
        ueig_vec_new = np.matmul(D1, Phi_vec_new)
        
        dupdx_new   = 1j*alpha*ueig_vec_new
        dvpdy_new   = np.matmul(D1, veig_vec_new)

        # normalizing u and v by max(abs(u))
        norm_u_new   = np.max(np.abs(ueig_vec_new))
        print("norm_u_new = ", norm_u_new)
        
        ueig_vec_new = ueig_vec_new/norm_u_new
        veig_vec_new = veig_vec_new/norm_u_new
        
        uv_eig      = ueig_vec*veig_vec
        uv_eig_new  = ueig_vec_new*veig_vec_new

        ## Renormalize to match Michalke
        Phi_vec_new  = Phi_vec_new/Phi_vec_new[mid_idx]
        
    flag1 = False
    
    if (plot_eigvcts and flag1):

        ptn = plt.gcf().number + 1
        
        fbbb = plt.figure(ptn)
        plt.plot(y, np.abs(veig_vec), 'k', label="abs(veig_vec)")
        plt.plot(y, np.abs(veig_conj_vec), 'r', label="abs(veig_conj_vec)")
        plt.xlabel('y')
        plt.ylabel('u')
        plt.title('u')
        plt.legend(loc="upper right")
        fbbb.show()
        
        ptn = ptn+1
        
        faaa = plt.figure(ptn)
        plt.plot(y, veig_vec.imag, 'k', label="p (imag)")
        plt.plot(y, v_from_pres.imag, 'r', label="p (imag)")
        plt.xlabel('y')
        plt.ylabel('v from pres')
        plt.title('v from pres')
        plt.legend(loc="upper right")
        faaa.show()
        
        ptn = ptn+1
        
        fac = plt.figure(ptn)
        plt.plot(y, updppdx.real + vpdppdy.real, 'k', label="p (real)")
        plt.plot(y, updppdx.imag + vpdppdy.imag, 'r', label="p (imag)")
        plt.xlabel('y')
        plt.ylabel('u''dp''/dx + v''dp''/dy')
        plt.title('u''dp''/dx + v''dp''/dy')
        plt.legend(loc="upper right")
        fac.show()
        
        ptn = ptn+1
        
        fab = plt.figure(ptn)
        plt.plot(y, Re_stress, 'k', label="Re stress")
        plt.plot(y, Re_stress_new, 'r', label="Imaginary part")
        #
        plt.xlabel('y')
        plt.ylabel('Reynold''s stress')
        plt.title('Reynold''s stress')
        plt.legend(loc="upper right")
        fab.show()
        
        ptn = ptn+1
        
        fa = plt.figure(ptn)
        plt.plot(y, ueig_vec.real, 'k', label="Real part")
        plt.plot(y, ueig_vec.imag, 'r', label="Imaginary part")
        plt.plot(y, np.abs(ueig_vec), 'b', label="Amplitude")
        plt.plot(y, ueig_vec_new.real, 'g--', label="Real (NEW)")
        plt.plot(y, ueig_vec_new.imag, 'c--', label="Imag (NEW)")
        plt.plot(y, np.abs(ueig_vec_new), 'm--', label="Amplitude (NEW)")
        #
        plt.xlabel('y')
        plt.ylabel('U-VEL')
        plt.title('U-VEL ONLY ===> CHECK')
        plt.legend(loc="upper left")
        fa.show()
        
        ptn = ptn+1
        
        faa = plt.figure(ptn)
        plt.plot(y, veig_vec.real, 'k', label="Real part")
        plt.plot(y, veig_vec.imag, 'r', label="Imaginary part")
        plt.plot(y, np.abs(veig_vec), 'b', label="Amplitude")
        plt.plot(y, veig_vec_new.real, 'g--', label="Real (NEW)")
        plt.plot(y, veig_vec_new.imag, 'c--', label="Imag (NEW)")
        plt.plot(y, np.abs(veig_vec_new), 'm--', label="Amplitude (NEW)")
        #
        plt.xlabel('y')
        plt.ylabel('V-VEL')
        plt.title('V-VEL ONLY ===> CHECK')
        plt.legend(loc="upper left")
        faa.show()
        
        ptn = ptn+1
        
        ff = plt.figure(ptn)
        #plt.plot(y, Phi_vec.real, 'r', label="Real part")
        #plt.plot(y, Phi_vec.imag, 'k', label="Imaginary part")
        plt.plot(y, Phi_vec_new.real, 'b--', label="Real part")
        plt.plot(y, Phi_vec_new.imag, 'g--', label="Imaginary part")
        #plt.plot(y, phase_vvel, 'm+', label="phase vvel")
        plt.xlabel('y')
        plt.ylabel('Phi')
        plt.title('Eigenfunction of Phi')
        plt.legend(loc="upper right")
        plt.xlim([0, 8])
        plt.ylim([0, 1.2])
        ff.show()
        
        #input("Press any key to continue")

    return ueig_vec, veig_vec, weig_vec, peig_vec, reig_vec


def read_input_file(inFile, bsfl):

    with open(inFile) as f:
        lines = f.readlines()

    nlines  = len(lines)

    # Solver type: 1 -> Rayleigh-Taylor, 2 -> Mixing-layer/Poiseuille solver
    i=0
    tmp = lines[i].split("#")[0]
    SolverT = int(tmp)
    
    # baseflowT: 1 -> hyperbolic-tangent baseflow, 2 -> plane Poiseuille baseflow
    i=i+1
    
    tmp = lines[i].split("#")[0]
    baseflowT = int(tmp)

    # ny
    i=i+1
    
    tmp = lines[i].split("#")[0]
    ny  = int(tmp)

    # Re_min, Re_max, npts_re
    i=i+1

    tmp = lines[i].split("#")[0]
    tmp = tmp.split(",")

    Re_min = float(tmp[0])
    Re_max = float(tmp[1])
    npts_re  = int(tmp[2])

    if (Re_min == Re_max): npts_re = 1

    #Re  = float(tmp)

    # Froude number, Schmidt number, reference velocity Uref and acceleration gref
    i=i+1

    tmp = lines[i].split("#")[0]
    tmp = tmp.split(",")

    bsfl.Fr   = float(tmp[0])
    bsfl.Sc   = float(tmp[1])
    bsfl.Uref = float(tmp[2])
    bsfl.gref = float(tmp[3])

    bsfl.Re_min = Re_min
    bsfl.Re_max = Re_max
    bsfl.npts_re = npts_re 

    # alpha_min, alpha_max and npts_alpha
    i=i+1

    tmp = lines[i].split("#")[0]
    tmp = tmp.split(",")
    
    alpha_min = float(tmp[0])
    alpha_max = float(tmp[1])
    npts_alp  = int(tmp[2])

    if (alpha_min == alpha_max): npts_alp = 1

    alpha = np.linspace(alpha_min, alpha_max, npts_alp)
    
    # beta
    i=i+1

    tmp = lines[i].split("#")[0]
    beta  = float(tmp)

    # yinf
    i=i+1

    tmp = lines[i].split("#")[0]
    yinf  = float(tmp)

    # lmap
    i=i+1

    tmp = lines[i].split("#")[0]
    lmap  = float(tmp)
    
    # target
    i=i+1

    tmp = lines[i].split("#")[0]
    tmp = tmp.split(",")

    target = float(tmp[0]) + 1j*float(tmp[1])
    
    print("")
    print("Reading input file")
    print("==================")
    print("SolverT                     = ", SolverT)
    print("baseflowT                   = ", baseflowT)
    print("ny                          = ", ny)
    print("Re_min                      = ", Re_min)
    print("Re_max                      = ", Re_max)
    print("npts_re                     = ", npts_re)
    print("Reference velocity Uref     = ", bsfl.Uref)
    print("Reference acceleration gref = ", bsfl.gref)
    print("Froude number               = ", bsfl.Fr)
    print("Schmidt number              = ", bsfl.Sc)
    print("alpha_min                   = ", alpha_min)
    print("alpha_max                   = ", alpha_max)
    print("npts_alp                    = ", npts_alp)
    print("beta                        = ", beta)
    print("yinf                        = ", yinf)
    print("lmap                        = ", lmap)
    print("target                      = ", target)
    print("==> Eigenfunction will be extracted for omega = ", target)

    print("")

    rt_flag = False

    if (SolverT==1):
        #print("Rayleigh-Taylor Eigenvalue Solver")
        rt_flag = True
    elif (SolverT==2):
        pass
        #print("Mixing-Layer/Poiseuille Eigenvalue Solver")
    else:
        sys.exit("Not a proper value for flag SolverT!!!!")
        
    return rt_flag, SolverT, baseflowT, ny, Re_min, Re_max, npts_re, alpha_min, alpha_max, npts_alp, alpha, beta, yinf, lmap, target


def compute_stream_function_check(ueig, veig, alpha, D1, y, ny):
    """
    This function compute the stream function from w-eigenfunction, then computes the u-eigenfunction from:
    u_eigenfct = D1*Phi_eigenfunction
    """

    Phi_eigfct = -veig/(1j*alpha)
    
    phase_Phi  = np.arctan2(Phi_eigfct.imag, Phi_eigfct.real)
    amp_Phi    = np.abs(Phi_eigfct)

    phase_Phi_uwrap = np.unwrap(phase_Phi)

    mid_idx   = get_mid_idx(ny)
    phase_ref = phase_Phi_uwrap[mid_idx]

    #print("phase_Phi_uwrap = ", phase_Phi_uwrap)
    #print("")
    #print("phase_ref = ", phase_ref)

    #plot_phase(phase_Phi_uwrap, "Phi before shift", y)
    phase_Phi_uwrap = phase_Phi_uwrap - phase_ref
    #plot_phase(phase_Phi_uwrap, "Phi after shift", y)

    Shift_Phi = 1
    if Shift_Phi == 1:
        Phi_eig = amp_Phi*np.exp(1j*phase_Phi_uwrap)
    else:
        Phi_eig = Phi_eigfct

    plot_real_imag_part(Phi_eig, "Phi", y)

    # Remember Thomas (1953): "the stability of plane Poiseuille flow"
    # Thomas uses the ansatz Psi = Phi(y)*exp[-i(alpha*x-omega*t)]
    # he uses -i instead of the more common i
    # Therefore:
    # ---> His eigenvalues will be complex conjugate of my eigenvalues
    # ---> His eigenvectors will be complex conjugate of my eigenvectors

    Phi_eig_cc = np.conj(Phi_eig)

    max_r = np.max(Phi_eig_cc.real)
    max_i = np.max(Phi_eig_cc.imag)

    ptn = plt.gcf().number + 1
    
    f1 = plt.figure(ptn)
    plt.plot(y, Phi_eig_cc.real, 'k', linewidth=1.5)
    plt.xlabel('y', fontsize=20)
    plt.ylabel(r'$\hat{\phi}_{cc, r}$', fontsize=20)
    plt.gcf().subplots_adjust(left=0.16)
    plt.gcf().subplots_adjust(bottom=0.16)
    plt.xlim([0, 1])
    plt.ylim([0, 1.02*max_r])
    #plt.gca().invert_xaxis()
    f1.show()

    ptn = ptn + 1
    
    f2 = plt.figure(ptn)
    plt.plot(y, Phi_eig_cc.imag, 'k', linewidth=1.5)
    plt.xlabel('y', fontsize=20)
    plt.ylabel(r'$\hat{\phi}_{cc, i}$', fontsize=20)
    plt.gcf().subplots_adjust(left=0.16)
    plt.gcf().subplots_adjust(bottom=0.16)
    plt.xlim([0, 1])
    plt.ylim([0, 1.02*max_i])
    #plt.gca().invert_xaxis()
    f2.show()

    Phiprime = np.matmul(D1, Phi_eig)

    # From Stuart paper 1957: "On the non-linear mechanics of hydrodynamic stability"
    Re_stress = 2.0*( Phi_eig.imag*Phiprime.real - Phi_eig.real*Phiprime.imag )

    # Here I am using u*v + uv*, where a star denotes complex conjugation
    Re_stress_222 = np.multiply(np.conj(ueig), veig ) + np.multiply(ueig, np.conj(veig) ) 

    ptn = ptn + 1
    
    f3 = plt.figure(ptn)
    plt.plot(y, Re_stress, 'k', linewidth=1.5, label="Stuart (1957) formula")
    plt.plot(y, Re_stress, 'r--', linewidth=1.5, label=r'$u^*v + uv^*$')
    plt.xlabel('y', fontsize=20)
    plt.ylabel("Reynolds stress", fontsize=20)
    plt.gcf().subplots_adjust(left=0.16)
    plt.gcf().subplots_adjust(bottom=0.16)
    plt.xlim([0, 1])
    plt.ylim([0, 0.04])
    plt.legend(loc="upper right")
    #plt.gca().invert_xaxis()
    f3.show()

    #ueig_fct_check = np.matmul(D1, Phi_eig)
    
    #plot_real_imag_part(ueig, "u", y)
    #plot_real_imag_part(ueig_fct_check, "u (indirectly from Phi)", y)

    #plot_amplitude(ueig, "u", y)
    #plot_amplitude(ueig_fct_check, "u (indirectly from Phi)", y)

    return Phi_eig_cc



def read_thomas_data(Phi_eig_cc, y_in, D1):

    # Number of header lines
    skip_header = 2

    with open("../data/Thomas_data.dat") as f:
        lines = f.readlines()

    nlines  = len(lines)-skip_header
    
    print("Thomas data (1953), number of lines in file :",nlines)

    # Wall-normla coordinate
    y = np.zeros(nlines, dp)

    # Stream-function eigenfunction
    phi = np.zeros(nlines, dpc)

    for i in range(0, nlines):
        tmp = lines[i+skip_header].split(",")
        y[i] = float(tmp[0])
        phi[i] = float(tmp[1]) + 1j*float(tmp[2])

    ptn = plt.gcf().number + 1

    Phi_eig_cc = Phi_eig_cc/np.max(Phi_eig_cc.real)

    # create figure and axis objects with subplots()
    fig,ax = plt.subplots()

    # make a plot
    ax.plot(y, phi.real, 'k', label=r"$\hat{\phi}_r$ (Thomas, 1953)")
    ax.plot(y_in, Phi_eig_cc.real, 'r', linestyle='--', dashes=(5, 5), label=r"$\hat{\phi}_r$ (Current)")
    # set x-axis label
    ax.set_xlabel("y", fontsize=20)
    # set y-axis label
    ax.set_ylabel(r"$\hat{\phi}_r$", fontsize = 20)
    # set axis limits
    ax.axis(xmin=0.,xmax=1.)
    ax.axis(ymin=0.,ymax=1.)

    # twin object for two different y-axis on the sample plot
    ax2=ax.twinx()
    # second plot
    ax2.plot(y, phi.imag, 'b', label=r"$\hat{\phi}_i$ (Thomas, 1953)")
    ax2.plot(y_in, Phi_eig_cc.imag, 'm', linestyle='--', dashes=(5, 5), label=r"$\hat{\phi}_i$ (Current)")
    # set x-axis label
    ax2.set_xlabel("y", fontsize=20)
    # set y-axis label
    ax2.set_ylabel(r"$\hat{\phi}_i$", fontsize = 20)
    # set axis limits
    ax2.axis(xmin=0.,xmax=1.)
    ax2.axis(ymin=0.,ymax=0.025)

    # ratio = 1.0
    # xleft, xright = ax2.get_xlim()
    # ybottom, ytop = ax2.get_ylim()
    # ax2.set_aspect(abs((xright-xleft)/(ybottom-ytop))*ratio)


    #ax2.legend(loc="upper right")

    plt.gcf().subplots_adjust(left=0.15)
    plt.gcf().subplots_adjust(right=0.80)
    plt.gcf().subplots_adjust(bottom=0.14)

    x_leg=0.18
    y_leg=0.25
    fig.legend(loc=(x_leg, y_leg))

    fig.show()
    plt.savefig("Phi_comparison_with_Thomas_data",bbox_inches='tight')


    # compute derivatives and Reynolds stress on Thomas data
    # Computing the Reynods stress from Thomas data using Stuart formula
    # Thomas (1953): "The stability of plane Poiseuille flow"
    # Stuart (1957): "On the non-linear mechanics of hydrodynamic stability"

    dphi_rdy = np.zeros(nlines, dp)
    dphi_idy = np.zeros(nlines, dp)
    
    c1 = -3./2.
    c2 = 2.
    c3 = -0.5

    dy = y[1]-y[0]
    
    dphi_rdy[0] = ( c1*phi[0].real + c2*phi[1].real + c3*phi[2].real )/dy
    dphi_idy[0] = ( c1*phi[0].imag + c2*phi[1].imag + c3*phi[2].imag )/dy

    dphi_rdy[-1] = ( c1*phi[nlines-1].real + c2*phi[nlines-2].real + c3*phi[nlines-3].real )/dy
    dphi_idy[-1] = ( c1*phi[nlines-1].imag + c2*phi[nlines-2].imag + c3*phi[nlines-3].imag )/dy
    
    for i in range(1, nlines-1):
        dphi_rdy[i] = ( phi[i+1].real - phi[i-1].real )/(2.*dy)
        dphi_idy[i] = ( phi[i+1].imag - phi[i-1].imag )/(2.*dy)

    # I added a minus here
    Re_stress_thomas_stuart = -2.0*( np.multiply(phi.imag, dphi_rdy) - np.multiply(phi.real, dphi_idy) )

    # compute derivatives and Reynolds stress on my data
    dPhi_eig_cc_rdy = np.matmul(D1, Phi_eig_cc.real) 
    dPhi_eig_cc_idy = np.matmul(D1, Phi_eig_cc.imag)

    # I added a minus here
    Re_stress       = -2.0*( np.multiply(Phi_eig_cc.imag, dPhi_eig_cc_rdy) - np.multiply(Phi_eig_cc.real, dPhi_eig_cc_idy) )

    f = plt.figure()
    ax = f.add_subplot(111)
    # I am p[lotting minus here for now =======> check
    plt.plot(y, Re_stress_thomas_stuart, 'k', label="Stuart (1957)")
    plt.plot(y_in, Re_stress, 'r', linestyle='--', dashes=(5, 5), label="Current")
    plt.xlabel('y', fontsize=18)
    plt.ylabel('Reynolds stress', fontsize=18)
    plt.legend(loc="upper left")
    plt.gcf().subplots_adjust(left=0.17)
    plt.gcf().subplots_adjust(bottom=0.15)
    plt.xlim([0, 1])
    plt.ylim([0, 0.2])    
    f.show()

    ratio = 0.70
    xleft, xright = ax.get_xlim()
    ybottom, ytop = ax.get_ylim()
    ax.set_aspect(abs((xright-xleft)/(ybottom-ytop))*ratio)

    plt.savefig("Reynolds_stress_comparison",bbox_inches='tight')
    

def compute_growth_rate_from_energy_balance(ueig, veig, peig, U, Uy, D1, y, alpha, Re):

    # Disturbance kinetic energy
    uu = compute_inner_prod(ueig, ueig)
    vv = compute_inner_prod(veig, veig)
    Et_integrand = 0.5*( uu + vv )
    Et = trapezoid_integration(Et_integrand, y)

    # Baseflow terms
    Et_Baseflow = 0.5*( np.multiply(U, U) )
    Dissip_baseflow = 1/Re*( np.multiply(Uy, Uy) )

    # Production
    uv = compute_inner_prod(ueig, veig)
    Production_integrand = -np.multiply(uv, Uy)

    #print("Production_integrand", Production_integrand)
    Production = trapezoid_integration(Production_integrand, y)

    # Dissipation
    term1 = compute_inner_prod(np.matmul(D1, ueig), np.matmul(D1, ueig))
    term2 = compute_inner_prod(1j*alpha*veig, 1j*alpha*veig)
    term3 = compute_inner_prod(np.matmul(D1, ueig), 1j*alpha*veig)
    
    Dissipation_integrand = -1/Re*( term1 + term2 - 2.0*term3 )
    Dissipation = trapezoid_integration(Dissipation_integrand, y)

    # Dissipation ----- Other way
    term1_other = 2.*compute_inner_prod(1j*alpha*ueig, 1j*alpha*ueig)
    term2_other = 2.*compute_inner_prod(np.matmul(D1, veig), np.matmul(D1, veig))
    term3_other = compute_inner_prod(np.matmul(D1, ueig), np.matmul(D1, ueig))
    term4_other = compute_inner_prod(1j*alpha*veig, 1j*alpha*veig)
    term5_other = 2.*compute_inner_prod(np.matmul(D1, ueig), 1j*alpha*veig)
    
    Dissipation_integrand_other = -1/Re*( term1_other + term2_other + term3_other + term4_other + term5_other )
    Dissipation_other = trapezoid_integration(Dissipation_integrand_other, y)

    print("OLD -----> Dissipation_other = ", Dissipation_other)

    # Dissipation ----- Other way 333
    term1_other333 = compute_inner_prod(1j*alpha*ueig, 1j*alpha*ueig)
    term2_other333 = compute_inner_prod(np.matmul(D1, ueig), np.matmul(D1, ueig))
    term3_other333 = compute_inner_prod(1j*alpha*veig, 1j*alpha*veig)
    term4_other333 = compute_inner_prod(np.matmul(D1, veig), np.matmul(D1, veig))

    Dissipation_integrand_other333 = -1/Re*( term1_other333 + term2_other333 + term3_other333 + term4_other333 )
    Dissipation_other333 = trapezoid_integration(Dissipation_integrand_other333, y)

    print("Dissipation, Dissipation_other, Dissipation_other333 = ",Dissipation, Dissipation_other, Dissipation_other333)

    omega_i = Production/(2.*Et) + Dissipation/(2.*Et)
    omega_i333 = Production/(2.*Et) + Dissipation_other333/(2.*Et)

    print("Production, Dissipation = ", Production, Dissipation)
    print("omega_i = ", omega_i)
    print("omega_i333 = ", omega_i333)

    if ( np.max(np.abs(Production_integrand.imag)) > 0.0 or np.max(np.abs(Dissipation_integrand.imag)) > 0.0 ):
        sys.exit("Production_integrand and/or Dissipation_integrand are not real!!!!!")

    # Compute pressure transport term
    Pres_trans_integrand = - ( compute_inner_prod(ueig, 1j*alpha*peig) + compute_inner_prod(veig, np.matmul(D1, peig)) )

    # Model term
    Cmu = 0.09
    # I need a minus I think because in eqution it is LHS = Production - Dissipation with dissipation > 0 and here above I used minus sign
    Mod_term = -Cmu*np.divide(np.multiply(Et_integrand, Et_integrand), Dissipation_integrand_other333)*np.multiply(Uy, Uy)


    #Et_Baseflow = 0.5*( np.multiply(U, U) )
    #Dissip_baseflow = 1/Re*( np.multiply(Uy, Uy) )

    eps_min = 1.e-8
    Uy2 = np.multiply(Uy, Uy) + eps_min

    k2_bsfl = np.multiply(Et_Baseflow, Et_Baseflow)
    eps2_bsfl = np.multiply(Dissip_baseflow, Dissip_baseflow) + eps_min

    Dissip_baseflow = Dissip_baseflow + 1.0e-8
    coef_prod = np.multiply(2.0*np.divide(Et_Baseflow, Dissip_baseflow), Uy2)
    coef_diss = np.multiply(np.divide(k2_bsfl, eps2_bsfl), Uy2)

    Mod_term_alt = Cmu*np.multiply(coef_prod, Et_integrand) - Cmu*np.multiply(coef_diss, Dissipation_integrand_other)

    #print("Dissip_baseflow = ", Dissip_baseflow)
    #print("Mod_term_alt = ", Mod_term_alt)
    #print("Mod_term_alt[0:5] = ", Mod_term_alt[0:5])
    #print("Mod_term_alt[-1] = ", Mod_term_alt[-1])
    #print("max(Mod_term_alt) = ", np.max(np.abs(Mod_term_alt)))

    # Plot production/dissipation integrand
    plot_production_dissipation(np.real(Production_integrand), np.real(Dissipation_integrand_other333), np.real(Pres_trans_integrand), np.real(Et_integrand), np.real(Mod_term), np.real(Mod_term_alt), y)

    plot_dissipation_only(np.real(Dissipation_integrand), np.real(Dissipation_integrand_other), np.real(Dissipation_integrand_other333), y)
    
def compute_inner_prod(f, g):

    inner_prod = 0.5*( np.multiply(np.conj(f), g) + np.multiply(f, np.conj(g)) )

    return inner_prod

    
def trapezoid_integration(fct, y):

    nn = len(y)
    wt = np.zeros(nn-1, dp)
    
    integral = 0.0 

    for i in range(0, nn-1):
       wt[i]  = ( y[i+1]-y[i] )/2.0 
       integral = integral + wt[i]*( fct[i] + fct[i+1] )

    return integral

def trapezoid_integration_cum(fct, y):

    nn = len(y)
    wt = np.zeros(nn-1, dp)
    
    integral = np.zeros(nn, dp)

    for i in range(0, nn-1):
       wt[i]       = ( y[i+1]-y[i] )/2.0
       integral[i+1] = integral[i] + wt[i]*( fct[i] + fct[i+1] )

    return integral

def plot_production_dissipation(prod, dissip, pres_trans, Et, Mod_term, Mod_term_alt, y):

    ymin = -10
    ymax = 10

    max_Mod_term = np.max(np.abs(Mod_term))
    max_prod = np.max(np.abs(prod))

    scaling = max_prod/max_Mod_term
    
    Mod_term = Mod_term*scaling

    ptn = plt.gcf().number + 1
    
    f = plt.figure(ptn)
    plt.plot(prod, y, 'k', linewidth=1.5)#, label="Production integrand")
    plt.xlabel(r"Production integrand: $-\left<\hat{u},\hat{v} \right> \bar{U}'$", fontsize=14)
    plt.ylabel('y', fontsize=16)
    plt.gcf().subplots_adjust(left=0.16)
    plt.gcf().subplots_adjust(bottom=0.15)
    plt.ylim([ymin, ymax])
    f.show()    

    ptn = ptn + 1
    
    f1 = plt.figure(ptn)
    plt.plot(dissip, y, 'k', linewidth=1.5)#, label="Dissipation integrand")
    plt.xlabel(r"Dissipation integrand: $-\frac{1}{Re}\left( \left<\mathcal{D}\hat{u},\mathcal{D}\hat{u}\right> + \left<i \alpha \hat{v}, i \alpha \hat{v}\right> - 2 \left< \mathcal{D}\hat{u}, i \alpha \hat{v} \right> \right)$", fontsize=14)
    plt.ylabel('y', fontsize=16)
    plt.gcf().subplots_adjust(left=0.16)
    plt.gcf().subplots_adjust(bottom=0.15)
    plt.ylim([ymin, ymax])
    f1.show()    

    ptn = ptn + 1
    
    f2 = plt.figure(ptn)
    plt.plot(pres_trans, y, 'k', linewidth=1.5)#, label="Dissipation integrand")
    plt.xlabel(r"Pressure transport integrand: $-\left( \left< \hat{u}, i \alpha \hat{p} \right> +  \left< \hat{v}, \mathcal{D}\hat{p}\right> \right)$", fontsize=14)
    plt.ylabel('y', fontsize=16)
    plt.gcf().subplots_adjust(left=0.16)
    plt.gcf().subplots_adjust(bottom=0.15)
    plt.ylim([ymin, ymax])
    f2.show()    

    ptn = ptn + 1
    
    f3 = plt.figure(ptn)
    plt.plot(prod, y, 'k', linewidth=1.5, label="Production")
    plt.plot(dissip, y, 'r', linewidth=1.5, label="Dissipation")
    plt.plot(pres_trans, y, 'b', linewidth=1.5, label="Pressure Transport")
    
    #label11 = "Model production ( scaling = %s )" % str('{:.2e}'.format(scaling))
    #plt.plot(Mod_term, y, 'g', linewidth=1.5, label=label11)
    
    plt.xlabel("Kinetic energy balance equation terms", fontsize=16)
    plt.ylabel('y', fontsize=16)
    
    plt.legend(loc="best")
    #plt.legend(bbox_to_anchor=(0.5, 0.98), loc='center', borderaxespad=0)
    plt.gcf().subplots_adjust(left=0.16)
    plt.gcf().subplots_adjust(bottom=0.15)
    #plt.ylim([ymin, ymax])
    #plt.xlim([-0.4, 0.8])
    plt.xlim([-0.14, 0.04])
    
    f3.show()


def plot_dissipation_only(dissip1, dissip2, dissip3, y):

    ymin = -10
    ymax = 10

    ptn = plt.gcf().number + 1
    
    f = plt.figure(ptn)
    plt.plot(dissip1, y, 'k', linewidth=1.5, label="Dissipation (1)")
    plt.plot(dissip2, y, 'r', linewidth=1.5, label="Dissipation (2)")
    plt.plot(dissip3, y, 'b--', linewidth=1.5, label="Dissipation (3)")
    plt.xlabel(r"Dissipation integrand", fontsize=14)
    plt.ylabel('y', fontsize=16)
    plt.gcf().subplots_adjust(left=0.16)
    plt.gcf().subplots_adjust(bottom=0.15)
    plt.legend(loc="best")
    plt.ylim([ymin, ymax])
    f.show()    

    input('Various dissipation functions')


# def write_eigvects_out(q, y, i, ny):

#     npts = q.shape[0]

#     idx_u = np.zeros(ny, i4)
#     idx_v = np.zeros(ny, i4)
#     idx_w = np.zeros(ny, i4)
#     idx_p = np.zeros(ny, i4)

#     for ii in range(0, ny):
#         idx_u[ii] = ii

#     idx_v = idx_u + 1*ny
#     idx_w = idx_u + 2*ny
#     idx_p = idx_u + 3*ny

#     # scaling factor
#     sf = np.max(np.abs(q[idx_u]))

#     q = q/sf

#     # variables are u, v, w, p
#     data_out = np.column_stack([ y, np.abs(q[idx_u]), np.abs(q[idx_v]), np.abs(q[idx_w]), np.abs(q[idx_p]), \
#                                  q[idx_u].real, q[idx_v].real, q[idx_w].real, q[idx_p].real, \
#                                  q[idx_u].imag, q[idx_v].imag, q[idx_w].imag, q[idx_p].imag                  ])
#     if (i==-666):
#         datafile_path = "./Eigenfunctions_Global_Solution.txt"
#     else:
#         datafile_path = "./Eigenfunctions_Local_Solution_" + str(i) + ".txt"
#     np.savetxt(datafile_path , data_out, fmt=['%21.11e','%21.11e','%21.11e','%21.11e','%21.11e','%21.11e','%21.11e','%21.11e','%21.11e','%21.11e','%21.11e','%21.11e','%21.11e'])



def write_eigvects_out_new(y, ueig, veig, weig, peig, reig, Local, bsfl_ref):

    ny = len(y)
    # variables are u, v, w, p
    # data_out = np.column_stack([ y, np.abs(q[idx_u]), np.abs(q[idx_v]), np.abs(q[idx_w]), np.abs(q[idx_p]), \
    #                              q[idx_u].real, q[idx_v].real, q[idx_w].real, q[idx_p].real, \
    #                              q[idx_u].imag, q[idx_v].imag, q[idx_w].imag, q[idx_p].imag                  ])

    data_out = np.column_stack([ y, np.abs(ueig), np.abs(veig), np.abs(weig), np.abs(peig), np.abs(reig), \
                                 ueig.real, veig.real, weig.real, peig.real, reig.real, \
                                 ueig.imag, veig.imag, weig.imag, peig.imag, reig.imag                  ])


    if (Local):
        datafile_path = "./Eigenfunctions_Local_Solution_nondimensional_" + str(ny) + "_pts.txt"
    else:
        datafile_path = "./Eigenfunctions_Global_Solution_nondimensional_" + str(ny) + "_pts.txt"

    np.savetxt(datafile_path , data_out, fmt=['%21.11e','%21.11e','%21.11e','%21.11e','%21.11e','%21.11e','%21.11e','%21.11e',\
                                              '%21.11e','%21.11e','%21.11e','%21.11e','%21.11e','%21.11e','%21.11e','%21.11e'])


    # Now write out dimensional eigenfunctions
    Lref = bsfl_ref.Lref
    Uref = bsfl_ref.Uref
    rhoref = bsfl_ref.rhoref

    ueig = ueig*Uref
    veig = veig*Uref
    weig = weig*Uref
    peig = peig*rhoref*Uref**2.
    reig = reig*rhoref

    ydim = y*Lref
    
    data_out = np.column_stack([ ydim, np.abs(ueig), np.abs(veig), np.abs(weig), np.abs(peig), np.abs(reig), \
                                 ueig.real, veig.real, weig.real, peig.real, reig.real, \
                                 ueig.imag, veig.imag, weig.imag, peig.imag, reig.imag                  ])


    if (Local):
        datafile_path = "./Eigenfunctions_Local_Solution_dimensional_" + str(ny) + "_pts.txt"
    else:
        datafile_path = "./Eigenfunctions_Global_Solution_dimensional_" + str(ny) + "_pts.txt"

    np.savetxt(datafile_path , data_out, fmt=['%21.11e','%21.11e','%21.11e','%21.11e','%21.11e','%21.11e','%21.11e','%21.11e',\
                                              '%21.11e','%21.11e','%21.11e','%21.11e','%21.11e','%21.11e','%21.11e','%21.11e'])

    
def set_local_flag_display_sizes(npts_alp, npts_re, plot_eigvcts):

    if ( npts_alp == 1 and npts_re == 1 ):
        print("Single alpha, single Reynolds ==> Global solution")
        print("")
        Local = False
        plot_eigvcts = 1
    else:
        Local = True
        if ( npts_alp > 1 ):
            if npts_re > 1:
                print("Mulitple alpha's and Re's")
            else:
                print("Mulitple alpha's, single Re")
        else:
            if npts_re > 1:
                print("Single alpha, multiple Re's")
            else:
                print("Single alpha, single Re")
            
    return Local, plot_eigvcts


def check_lmap_val_reset_if_needed(Var, map, lmap, ny):

    flag_reset = False

    for i in range( ny-1, -1, -1):
        
        if ( np.abs(Var[i]) > 1.0e-5 ):
            #print("map.y[i] = ", map.y[i])
            #print("bsfl.Up[i] = ", bsfl.Up[i])
            lmap_tmp = map.y[i]
            break

    quot = np.abs((lmap-lmap_tmp)/lmap)
    if ( quot > 0.005 ):
        warnings.warn("Resetting lmap to: %21.11e" % lmap_tmp)
        lmap = lmap_tmp
        flag_reset = True

    return flag_reset


def write_stability_banana(npts_alp, npts_re, iarr, alpha, Re_range, filename):

    work_dir      = "./"
    save_path     = work_dir #+ '/' + folder1
    completeName  = os.path.join(save_path, filename)
    fileoutFinal  = open(completeName,'w')
    
    zname2D       = 'ZONE T="2-D Zone", I = ' + str(npts_re) + ', J = ' + str(npts_alp) + '\n'
    fileoutFinal.write('TITLE     = "2D DATA"\n')
    fileoutFinal.write('VARIABLES = "X" "Y" "omega_r" "omega_i"\n')
    fileoutFinal.write(zname2D)
    
    for j in range(0, npts_alp):
        for i in range(0, npts_re):
            fileoutFinal.write("%25.15e %25.15e %25.15e %25.15e \n" % ( Re_range[i], alpha[j], iarr.omega_array[i,j].real, iarr.omega_array[i,j].imag ) )

    fileoutFinal.close()


def extrapolate_in_alpha(iarr, alpha, i, ire):

    om1   = iarr.omega_array[ire, i-1]
    om2   = iarr.omega_array[ire, i-2]
    al1   = alpha[i-1]
    al2   = alpha[i-2]
    
    acoef = ( om1 - om2 )/( al1 - al2 )
    bcoef = om1 - acoef*al1
    
    omega = acoef*alpha[i] + bcoef

    #print("alpha[i] = ",alpha[i])
    #print("acoef*alpha[i] = ", acoef*alpha[i])
    #print("om1, om2, al1, al2, acoef, bcoef, omega = ", om1, om2, al1, al2, acoef, bcoef, omega)

    return omega

def extrapolate_in_reynolds(iarr, i, ire):

    print("Reynolds extrapol")
    om1   = iarr.omega_array[ire-1, i]
    om2   = iarr.omega_array[ire-2, i]
    re1   = iarr.re_array[ire-1]
    re2   = iarr.re_array[ire-2]
    
    acoef = ( om1 - om2 )/( re1 - re2 )
    bcoef = om1 - acoef*re1
    
    omega = acoef*iarr.re_array[ire] + bcoef

    return omega

def comp_ke_bal_rt_boussinesq(ueig, veig, weig, peig, reig, map, mob, bsfl, D1, D2, y, alpha, beta, Re, omega_i, bsfl_ref, rt_flag, riist, Local):

    # Kinetic energy balance for Rayleigh-Taylor in the Boussinesq limit

    # This is non-dimensional

    if ( mob.boussinesq == 1 ):
        
        ny = len(ueig)

        Fr2 = bsfl_ref.Fr**2.
        
        # Get some quantities to compute the balances
        Mu, Mup, Rho, Rhop, Rhopp, rho2, rho3, rho_inv, rho2_inv, rho3_inv = get_baseflow_and_derivatives(bsfl, mob)
        ia, ib, ia2, ib2, iab, Du, Dv, Dw, Dr, D2u, D2v, D2w, D2r = get_eigenfunctions_quantities(alpha, beta, D1, D2, ueig, veig, weig, reig)
        
        ke = compute_disturbance_kin_energy(ueig, veig, weig)
        
        vpc1, vpc2, vpc3, vpc_tot = compute_velocity_pressure_correlations(ueig, veig, weig, peig, alpha, beta, D1)

        #
        # Compute LHS
        #
        
        LHS = 2*omega_i*ke

        #
        # Compute RHS
        #

        # 1) inviscid terms
        RHS_i1 = vpc_tot

        # Gravity Production term
        RHS_i2 = -1/Fr2*compute_inner_prod(reig, weig)

        # 2) viscous transport: d/dx_j(u'_i tau'_ij)
        RHS_v1 = compute_disturbance_viscous_transport(ueig, veig, weig, peig, reig, Re, alpha, beta, D1)

        # 3) viscous disturbance dissipation
        epsilon_dist = compute_disturbance_dissipation(ueig, veig, weig, peig, reig, y, alpha, beta, D1, D2, bsfl, bsfl_ref, mob)[1]

        RHS = RHS_i1 + RHS_i2 + RHS_v1 - epsilon_dist

        print("")
        print("KINETIC ENERGY EQUATION BALANCE FOR BOUSSINESQ")
        print("----------------------------------------------")
        energy_bal = np.amax(np.abs(LHS-RHS))
        print("Kinetic Energy balance (RT) = ", energy_bal)
        print("")
        
        #
        # Plot dissipation components
        #

        ptn = plt.gcf().number + 1

        # Get ymin-ymax
        ymin, ymax = FindResonableYminYmax(RHS_i2, y)
        
        f  = plt.figure(ptn)
        ax = plt.subplot(111)

        ax.plot(np.real(LHS), y, 'k', linewidth=1.5, label=r"LHS")
        ax.plot(np.real(RHS), y, 'r--', linewidth=1.5, label=r"RHS")
        
        ax.plot(np.real(RHS_i1), y, 'b', linewidth=1.5, label=r"$vp: -u'_i \dfrac{\partial p'}{\partial x_i}$ (RHS)")

        # Gravity production
        ax.plot(np.real(RHS_i2), y, 'c', linewidth=1.5, label=r"gp: $-\dfrac{1}{Fr^2}\rho'w'g$ (RHS)")
        # Viscous transport
        ax.plot(np.real(RHS_v1), y, 'g', linewidth=1.5, label=r"vt: $\dfrac{\partial}{\partial x_j}(u'_i \tau'_{ij})$ (RHS)")
        # Viscous dissipation
        ax.plot(np.real(epsilon_dist), y, 'm', linewidth=1.5, label=r"vd: $\tau'_{ij}\dfrac{\partial u'_i}{\partial x_j}$ (RHS)")

        plt.xlabel("Boussinesq equations: Kinetic Energy balance (RT)", fontsize=14)
        plt.ylabel("z", fontsize=14)

        plt.gcf().subplots_adjust(left=0.17)
        plt.gcf().subplots_adjust(bottom=0.15)
        plt.ylim([ymin, ymax])
        
        plt.legend(loc="upper right")
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15),   # box with lower left corner at (x, y)
                  ncol=3, fancybox=True, shadow=True, fontsize=10)
        
        f.show()
        

def compute_terms_rayleigh_taylor(ueig, veig, weig, peig, reig, map, mob, bsfl, D1, D2, y, alpha, beta, Re, omega_i, bsfl_ref, rt_flag, riist, Local):

    #plt.close('all') ==> this command freezes my machine

    ny = len(ueig)
    
    print("")
    print("Closing all previous plots...")
    close_previous_plots()
        
    print("")
    print("Kinetic Energy balance plots and calculations")
    print("=============================================")
    print("")

    print("Re = ", Re)

    Sc = bsfl_ref.Sc
    Re = bsfl_ref.Re
    Fr2 = bsfl_ref.Fr**2.

    gref = bsfl_ref.gref
    Lref = bsfl_ref.Lref
    Uref = bsfl_ref.Uref
    rhoref = bsfl_ref.rhoref
    muref  = bsfl_ref.muref
    
    ny   = len(y)
    yinf = riist.yinf

    # Get some quantities to compute the balances
    Mu, Mup, Rho, Rhop, Rhopp, rho2, rho3, rho_inv, rho2_inv, rho3_inv = get_baseflow_and_derivatives(bsfl, mob)
    ia, ib, ia2, ib2, iab, Du, Dv, Dw, Dr, D2u, D2v, D2w, D2r = get_eigenfunctions_quantities(alpha, beta, D1, D2, ueig, veig, weig, reig)

    dudx,dudy,dudz,dvdx,dvdy,dvdz,dwdx,dwdy,dwdz,dpdx,dpdy,dpdz,drdx,drdy,drdz = get_eigenfunction_derivatives(ueig, veig, weig, peig, reig, alpha, beta, D1)
    
    d2wdx2  = (1j*alpha)**2.*weig
    d2wdy2  = (1j*beta)**2.*weig
    d2wdz2  = D2w

    d2udx2  = (1j*alpha)**2.*ueig
    d2udy2  = (1j*beta)**2.*ueig
    d2udz2  = D2u


    #####################################################################
    # KINETIC ENERGY EQUATION
    #####################################################################
    
    # Disturbance kinetic energy
    Et_integrand = compute_disturbance_kin_energy(ueig, veig, weig)
    
    # Velocity pressure correlation
    vp1 = -compute_inner_prod(ueig, 1j*alpha*peig)
    vp2 = -compute_inner_prod(veig, 1j*beta*peig)
    vp3 = -compute_inner_prod(weig, np.matmul(D1, peig))

    VelPres = vp1 + vp2 + vp3

    #minus_dpu_idxj = 
    
    if ( mob.boussinesq == -2 ):

        # Compute the various terms non-dimensional too
        alpha_nd   = alpha*Lref
        beta_nd    = beta*Lref

        # Nondimensionalize eigenfunctions
        ueig_nd = ueig/Uref
        veig_nd = veig/Uref
        weig_nd = weig/Uref

        peig_nd = peig/(rhoref*Uref**2.)
        reig_nd = reig/rhoref

        norm_s, ueig_nd, veig_nd, weig_nd, peig_nd, reig_nd = normalize_eigenvectors(ueig_nd, veig_nd, weig_nd, peig_nd, reig_nd, rt_flag)
        
        vp1_nd = -compute_inner_prod(ueig_nd, 1j*alpha_nd*peig_nd)
        vp2_nd = -compute_inner_prod(veig_nd, 1j*beta_nd*peig_nd)
        vp3_nd = -compute_inner_prod(weig_nd, np.matmul(map.D1_nondim, peig_nd))

        VelPres_nd = vp1_nd + vp2_nd + vp3_nd

        ia_nd  = ( 1j*alpha_nd ) 
        ib_nd  = ( 1j*beta_nd )

        ia2_nd = ( 1j*alpha_nd )**2. 
        ib2_nd = ( 1j*beta_nd )**2.

        Du_nd = np.matmul(map.D1_nondim, ueig_nd)
        Dv_nd = np.matmul(map.D1_nondim, veig_nd)
        Dw_nd = np.matmul(map.D1_nondim, weig_nd)

        D2u_nd = np.matmul(map.D2_nondim, ueig_nd)
        D2v_nd = np.matmul(map.D2_nondim, veig_nd)
        D2w_nd = np.matmul(map.D2_nondim, weig_nd)
        D2r_nd = np.matmul(map.D2_nondim, reig_nd)
        
        visc1_nd  = compute_inner_prod(ia2_nd*ueig_nd, ueig_nd) + compute_inner_prod(ib2_nd*ueig_nd, ueig_nd) + compute_inner_prod(D2u_nd, ueig_nd)
        visc2_nd  = compute_inner_prod(ia2_nd*veig_nd, veig_nd) + compute_inner_prod(ib2_nd*veig_nd, veig_nd) + compute_inner_prod(D2v_nd, veig_nd)
        visc3_nd  = compute_inner_prod(ia2_nd*weig_nd, weig_nd) + compute_inner_prod(ib2_nd*weig_nd, weig_nd) + compute_inner_prod(D2w_nd, weig_nd)

    visc1  = compute_inner_prod(ia2*ueig, ueig) + compute_inner_prod(ib2*ueig, ueig) + compute_inner_prod(D2u, ueig)
    visc2  = compute_inner_prod(ia2*veig, veig) + compute_inner_prod(ib2*veig, veig) + compute_inner_prod(D2v, veig)
    visc3  = compute_inner_prod(ia2*weig, weig) + compute_inner_prod(ib2*weig, weig) + compute_inner_prod(D2w, weig)

    if   ( mob.boussinesq == 1 ):
        visc_t = np.divide(visc1 + visc2 + visc3, Re)
    elif ( mob.boussinesq == -2 ):
        visc_t = np.multiply(Mu, visc1 + visc2 + visc3)

        # The following three lines do not lead to the proper non-dimensional term
        # because the equation is nondimensionalized as a whole with the Reynolds
        # number appearing
        #scal_visc = muref*Uref**2/Lref**2.
        #visc_t_nd = np.multiply(bsfl.Mu_nd, visc1_nd + visc2_nd + visc3_nd)
        #visc_t_nd = visc_t/scal_visc
        visc_t_nd = np.divide(visc1_nd + visc2_nd + visc3_nd, Re)
        
        # Viscous (proportional to d mu_baseflow/dz)
        visc_grad1  = compute_inner_prod(ueig, Du) + compute_inner_prod(veig, Dv) + compute_inner_prod(weig, Dw)
        visc_grad2  = compute_inner_prod(ueig, ia*weig) + compute_inner_prod(veig, ib*weig) + compute_inner_prod(weig, Dw)
        visc_grad_t = np.multiply(Mup, visc_grad1 + visc_grad2)        

        # Need to check this by computing it for solver -3
        visc_grad1_nd  = compute_inner_prod(ueig_nd, Du_nd) + compute_inner_prod(veig_nd, Dv_nd) + compute_inner_prod(weig_nd, Dw_nd)
        visc_grad2_nd  = compute_inner_prod(ueig_nd, ia_nd*weig_nd) + compute_inner_prod(veig_nd, ib_nd*weig_nd) + compute_inner_prod(weig_nd, Dw_nd)
        # The following line does not seem to work for the same reason as above probably 
        #visc_grad_t_nd = np.multiply(bsfl.Mup_nd, visc_grad1_nd + visc_grad2_nd)
        visc_grad_t_nd = np.divide(visc_grad1_nd + visc_grad2_nd, Re)        

    elif ( mob.boussinesq == -3 or mob.boussinesq == -4 ):
        visc1_3Re = compute_inner_prod(ueig, ia2*ueig) + compute_inner_prod(veig, ib2*veig) + compute_inner_prod(weig, D2w)
        visc2_3Re = compute_inner_prod(ueig, iab*veig) + compute_inner_prod(veig, iab*ueig) + compute_inner_prod(weig, ia*Du)
        visc3_3Re = compute_inner_prod(ueig, ia*Dw) + compute_inner_prod(veig, ib*Dw) + compute_inner_prod(weig, ib*Dv)

        visc_t = np.divide(visc1 + visc2 + visc3, Re)
        visc_t_3Re = np.divide(visc1_3Re + visc2_3Re + visc3_3Re, 3*Re)

        if (mob.boussinesq == -4):

            visc_t = np.multiply(Mu, visc_t)
            visc_t_3Re = np.multiply(Mu, visc_t_3Re)

            # Viscous (proportional to d mu_baseflow/dz)
            visc_grad1  = compute_inner_prod(ueig, Du) + compute_inner_prod(veig, Dv) + compute_inner_prod(weig, Dw)
            visc_grad2  = compute_inner_prod(ueig, ia*weig) + compute_inner_prod(veig, ib*weig) + compute_inner_prod(weig, Dw)
            visc_grad3  = -2./3.*( compute_inner_prod(weig, ia*ueig) + compute_inner_prod(weig, ib*veig) + compute_inner_prod(weig, Dw) )
            visc_grad_t = np.multiply(Mup, visc_grad1 + visc_grad2 + visc_grad3 )/Re        
        
    # Build LHS and RHS
    if   ( mob.boussinesq == 1 ):
        # Gravity production term
        grav_prod = -compute_inner_prod(reig, weig)/Fr2 # Nondimensional here

        MultiplyThroughByRhoBaseflow = False
        if MultiplyThroughByRhoBaseflow:
            Et_integrand = np.multiply(Rho, Et_integrand)
            VelPres = np.multiply(Rho, VelPres)
            visc_t = np.multiply(Rho, visc_t)
            grav_prod = np.multiply(Rho, grav_prod)
        
        # Left-hand-side
        LHS = Et_integrand*2*omega_i
        # Right-hand-side
        RHS = VelPres + visc_t + grav_prod

        #RHS = np.multiply(Rho, RHS)
        #LHS = np.multiply(Rho, LHS)
        
    elif ( mob.boussinesq == -2 ):
        # Gravity production term
        grav_prod = -gref*compute_inner_prod(reig, weig)    # dimensional
        grav_prod_nd = -compute_inner_prod(reig_nd, weig_nd)/Fr2    # non-dimensional
        
        # Left-hand-side
        LHS = np.multiply(Et_integrand, Rho)*2*omega_i
        # Right-hand-side
        RHS = VelPres + visc_t + visc_grad_t + grav_prod
    elif ( mob.boussinesq == -3 or mob.boussinesq == -4 ):
        # Gravity production term
        grav_prod = -compute_inner_prod(reig, weig)/Fr2 # Nondimensional here
        # Left-hand-side
        LHS = np.multiply(Et_integrand, Rho)*2*omega_i
        # Right-hand-side
        RHS = VelPres + grav_prod + visc_t_3Re + visc_t

        if (mob.boussinesq == -4):
            RHS = RHS + visc_grad_t 
        
    ##########################################
    # CHECK ALL THIS!!!!!!!!!!!!!!!!!!!!!!!!!
    ##########################################
    
    if   ( mob.boussinesq == 1 ):
        Et_int = trapezoid_integration(np.multiply(np.real(Et_integrand), 2), y)
    elif ( mob.boussinesq == -2 ):
        Et_int = trapezoid_integration(np.multiply(np.real(Et_integrand), 2*Rho), y)
    elif ( mob.boussinesq == -3 or mob.boussinesq == -4 ):
        Et_int = trapezoid_integration(np.multiply(np.real(Et_integrand), 2*Rho), y)
        
    VelPres_int = trapezoid_integration(np.real(VelPres), y)
    visc_t_int  = trapezoid_integration(np.real(visc_t), y)
    
    if   ( mob.boussinesq == -2 ):
        visc_grad_t_int = trapezoid_integration(np.real(visc_grad_t), y)
    elif ( mob.boussinesq == -3 or mob.boussinesq == -4 ):
        visc_t_3Re_int = trapezoid_integration(np.real(visc_t_3Re), y)

    grav_prod_int   = trapezoid_integration(np.real(grav_prod), y)

    if   ( mob.boussinesq == 1 ):
        omega_i_balance = ( VelPres_int + visc_t_int + grav_prod_int )/Et_int
    elif   ( mob.boussinesq == -2 ):
        omega_i_balance = ( VelPres_int + visc_t_int + visc_grad_t_int + grav_prod_int )/Et_int
    elif   ( mob.boussinesq == -3 or mob.boussinesq == -4 ):
        omega_i_balance = ( VelPres_int + visc_t_int + visc_t_3Re_int + grav_prod_int )/Et_int

    WriteThisInfo=False
    if   ( ( mob.boussinesq == -3 or mob.boussinesq == -4 ) and WriteThisInfo ):
        print("VelPres_int    = ", VelPres_int)
        print("visc_t_int     = ", visc_t_int)
        print("visc_t_3Re_int = ", visc_t_3Re_int)
        print("grav_prod_int  = ", grav_prod_int)
        print("Et_int         =", Et_int)
 
    print("")
    print("Comparison between growth rate from eigenvalue and energy balance")
    print("-----------------------------------------------------------------")
    if ( mob.boussinesq == -2 ):
        print("omega_i (dimensional)     = ", omega_i)
        print("omega_i (non-dimensional) = ", omega_i*bsfl_ref.Lref/bsfl_ref.Uref)
    else:
        print("omega_i (dimensional)     = ", omega_i*bsfl_ref.Uref/bsfl_ref.Lref)    
        print("omega_i (non-dimensional) = ", omega_i)    
    print("omega_i_balance = ", omega_i_balance)
    print("")
    
    #####################################################################
    # Plot 
    #####################################################################

    print("LHS and RHS should be real:")
    print("---------------------------")
    print("max(abs(imag(LHS))) = ", np.amax(np.abs(np.imag(LHS))))
    print("max(abs(imag(RHS))) = ", np.amax(np.abs(np.imag(RHS))))

    print("")
    print("Check energy balance: KINETIC ENERGY EQUATION")
    print("---------------------------------------------")
    energy_bal = np.amax(np.abs(LHS-RHS))
    print("Energy balance (RT) = ", energy_bal)
    print("")
    
    # Write data out
    if ( mob.boussinesq < 0 ):
        str_b = 'minus'
    else:
        str_b = 'plus'

    alpha_out = float(alpha)
    #print("type(alpha_out) = ", type(alpha_out))
    #print("type(alpha) = ", type(alpha))
    #print("{:.2f}".format(3.1234567890123))
    str_alp = "{:.4f}".format(alpha_out)
    str_bet = "{:.4f}".format(beta)

    #print("type(str_o) = ", type(str_o))

    if (Local):
        str_loc = 'Local'
    else:
        str_loc = 'Global'
        
    filename = 'Energy_balance_' + str_loc + '_alpha_' + str_alp + '_beta_' + str_bet + '_boussi_' + str_b + '_' + str(abs(mob.boussinesq)) + '.txt'
    
    if ( mob.boussinesq == -2 ):
        write_energy_balance_terms(ny , map.y/Lref, np.real(VelPres_nd), np.real(visc_t_nd), np.real(grav_prod_nd), filename)
    else:
        write_energy_balance_terms(ny, map.y, np.real(VelPres), np.real(visc_t), np.real(grav_prod), filename)

    # Dimensional and non-dimensional coordinates

    fac_lim = 1
    if ( mob.boussinesq == -2 ):
        zdim = y
        znondim = y/Lref
        
        zmaxdim = yinf/fac_lim
        zmaxnondim = yinf/Lref/fac_lim

        zmax = zmaxdim
        zmin = -zmax

    else:
        zdim = y*Lref
        znondim = y
        
        zmaxdim = yinf*Lref/fac_lim
        zmaxnondim = yinf/fac_lim

        zmax = zmaxnondim
        zmin = -zmax

    zmindim = -zmaxdim
    zminnondim = -zmaxnondim
    
    SetMinMaxPlot = True

    ptn = plt.gcf().number + 1

    if ( mob.boussinesq == -2 ):

        # Plot non-dimensionalized data from solver boussinesq == -2        
        f = plt.figure(ptn)
        plt.plot(np.real(visc_t_nd), znondim, 'g', linewidth=1.5, label=r"Viscous ($\bar{\mu}$) (non-dim)")
        plt.plot(np.real(visc_grad_t_nd), znondim, 'm', linewidth=1.5, label=r"Viscous ($d\bar{\mu}/dz$) (CHECK THIS !!!!!!!!!!!)")
        plt.plot(np.real(VelPres_nd), znondim, 'b', linewidth=1.5, label=r"Velocity-Pressure correlation (non-dim)")
        plt.plot(np.real(grav_prod_nd), znondim, 'c', linewidth=1.5, label=r"Gravity Production (non-dim)")
        plt.xlabel("Energy balance: Non-dimensional", fontsize=18)
        plt.ylabel("z (non-dim)", fontsize=18)
        plt.title('Solver boussinesq == -2') 
        
        plt.gcf().subplots_adjust(left=0.18)
        plt.gcf().subplots_adjust(bottom=0.15)
        #plt.xlim([-0.01, 0.01])
        if (SetMinMaxPlot):
            plt.ylim([zminnondim, zmaxnondim])
            f.show()
        plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper right', fontsize=8)

        ptn = ptn + 1


    # f = plt.figure(ptn)
    # plt.plot(np.real(LHS), y, 'k', linewidth=1.5, label=r"LHS")
    # plt.plot(np.real(RHS), y, 'r--', linewidth=1.5, label=r"RHS")
    # plt.xlabel("Energy balance R-T", fontsize=18)
    # plt.ylabel("y", fontsize=18)
    # plt.gcf().subplots_adjust(left=0.17)
    # plt.gcf().subplots_adjust(bottom=0.15)
    # #plt.xlim([0.0, 1.e-5])
    # if (SetMinMaxPlot):
    #     plt.ylim([zmin, zmax])
    # f.show()
    
    # plt.legend(loc="upper center")
    
    # ptn = ptn + 1




    
    # # Get ymin-ymax
    # ymin, ymax = FindResonableYminYmax(grav_prod, y)
        
    # f = plt.figure(ptn)
    # plt.plot(np.real(LHS), y, 'k', linewidth=1.5, label=r"LHS")
    # plt.plot(np.real(RHS), y, 'r--', linewidth=1.5, label=r"RHS")

    
    # plt.plot(np.real(VelPres), y, 'b', linewidth=1.5, label=r"$-u'_i \dfrac{\partial p'}{\partial x_i}$ (RHS)")
    # plt.plot(np.real(visc_t), y, 'g', linewidth=1.5, label=r"Viscous ($\bar{\mu}$) (RHS)")
    # if   ( mob.boussinesq == -2 ):
    #     plt.plot(np.real(visc_grad_t), y, 'm', linewidth=1.5, label=r"Viscous ($d\bar{\mu}/dz$) (RHS)")
    # plt.plot(np.real(grav_prod), y, 'c', linewidth=1.5, label=r"Gravity Production (RHS)")
    # plt.xlabel("Kinetic Energy balance components R-T", fontsize=14)
    # plt.ylabel("z", fontsize=14)
    
    # plt.gcf().subplots_adjust(left=0.18)
    # plt.gcf().subplots_adjust(bottom=0.15)

    # xmin = -25.
    # xmax = 45.
    # plt.xlim([xmin, xmax])
    
    # #ymin = -0.05
    # #ymax = -ymin
    # plt.ylim([ymin, ymax])
    
    # #if (SetMinMaxPlot):
    # #   plt.ylim([zmin, zmax]) 
    # f.show()

    # plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper right', fontsize=8)

    
    
    # ptn = ptn + 1
    
    # f = plt.figure(ptn)
    # plt.plot(np.real(visc_t), y, 'k', linewidth=1.5, label=r"visc_t")
    # plt.plot(np.real(compute_inner_prod(D2u, ueig)), y, 'r--', linewidth=1.5, label=r"D2u")
    # plt.plot(np.real(compute_inner_prod(D2v, veig)), y, 'c--', linewidth=1.5, label=r"D2v")
    # plt.plot(np.real(compute_inner_prod(D2w, weig)), y, 'g--', linewidth=1.5, label=r"D2w")
    # plt.xlabel("visc_t only + D2u + D2v + D2w", fontsize=18)
    # plt.ylabel("y", fontsize=18)
    # plt.gcf().subplots_adjust(left=0.17)
    # plt.gcf().subplots_adjust(bottom=0.15)
    # #plt.xlim([0.0, 1.e-5])
    # if (SetMinMaxPlot):
    #     plt.ylim([zmin, zmax])
    # f.show()
    
    # plt.legend(loc="upper center")

    
    # Check and plot growth rate as a function of z

    #print("RHS",RHS)
    #print("")
    #print("")
    #print("LHS",LHS)
    #print("")
    #print("")
    #print("LHS/RHS = ", np.divide(LHS,RHS))
    
    #growth_rate = np.divide(RHS, LHS)

    #ptn = ptn + 1

    # f = plt.figure(ptn)
    # plt.plot(growth_rate, y, 'k', linewidth=1.5, label=r"growth rate")
    # plt.xlabel("Energy balance R-T: growth rate vs. z", fontsize=18)
    # plt.ylabel("z", fontsize=18)

    # plt.gcf().subplots_adjust(left=0.17)
    # plt.gcf().subplots_adjust(bottom=0.15)
    # f.show()

    # plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper right', fontsize=8)
    

    ptn = ptn + 1
    
    # Plot velocity pressure correlation components

    Check_vp1 = np.max(np.abs(vp1.imag))
    Check_vp2 = np.max(np.abs(vp2.imag))
    Check_vp3 = np.max(np.abs(vp3.imag))
    Check_VelPres = np.max(np.abs(VelPres.imag))

    tol = 1e-20
    
    if ( Check_vp1 < tol and Check_vp2 < tol and Check_vp3 < tol and Check_VelPres < tol ):
        pass
    else:
        sys.exit("Error: some quantities have a nonzero imaginary part -- vp1, vp2, vp3, VelPres")

    # Density-velocity correlation derivative
    az = np.divide(compute_inner_prod(reig, weig), Rho)
    
    LaplacR11 = 2.* ( compute_inner_prod(dudx, dudx) + compute_inner_prod(dudy, dudy) + compute_inner_prod(dudz, dudz) )
    LaplacR11 = LaplacR11 + 2.* ( compute_inner_prod(ueig, d2udx2) + compute_inner_prod(ueig, d2udy2) + compute_inner_prod(ueig, d2udz2) )

    LaplacR33 = 2.* ( compute_inner_prod(dwdx, dwdx) + compute_inner_prod(dwdy, dwdy) + compute_inner_prod(dwdz, dwdz) )
    LaplacR33 = LaplacR33 + 2.* ( compute_inner_prod(weig, d2wdx2) + compute_inner_prod(weig, d2wdy2) + compute_inner_prod(weig, d2wdz2) )

    LaplacR13 = 2.* ( compute_inner_prod(dudx, dwdx) + compute_inner_prod(dudy, dwdy) + compute_inner_prod(dudz, dwdz) )
    LaplacR13 = LaplacR13 + ( compute_inner_prod(ueig, d2wdx2) + compute_inner_prod(ueig, d2wdy2) + compute_inner_prod(ueig, d2wdz2) )
    LaplacR13 = LaplacR13 + ( compute_inner_prod(weig, d2udx2) + compute_inner_prod(weig, d2udy2) + compute_inner_prod(weig, d2udz2) )

    LaplacTotal = LaplacR11 + LaplacR33 + LaplacR13

    maxR11 = np.amax(np.abs(LaplacR11))
    maxR33 = np.amax(np.abs(LaplacR33))
    maxR13 = np.amax(np.abs(LaplacR13))

    maxR11R33R13 = np.amax(np.abs(LaplacTotal))

    maxVelPres = np.amax(np.abs(VelPres))

    fac11 = maxVelPres/maxR11
    fac33 = maxVelPres/maxR33
    fac13 = maxVelPres/maxR13

    factot = maxVelPres/maxR11R33R13

    print("fac11, fac33, fac13 = ", fac11, fac33, fac13)

    # Compute also az*dp_baseflow/dz
    azdpbardz = np.multiply(az, Rho)*gref/Fr2

    maxazdpbardz = np.amax(np.abs(azdpbardz))
    fac_azdpbardz = maxVelPres/maxazdpbardz


    
    ptn = plt.gcf().number + 1

    ymin, ymax = FindResonableYminYmax(LaplacR33, y)
    
    f  = plt.figure(ptn)
    ax = plt.subplot(111)
    ax.plot(np.real(vp1), y, 'k', linewidth=1.5, label=r"$-\overline{u' \dfrac{\partial p'}{\partial x}}$")
    ax.plot(np.real(vp2), y, 'r--', linewidth=1.5, label=r"$-\overline{v' \dfrac{\partial p'}{\partial y}}$")
    ax.plot(np.real(vp3), y, 'b', linewidth=1.5, label=r"$-\overline{w' \dfrac{\partial p'}{\partial z}}$")
    #ax.plot(np.real(d2azdz2)/1000, y, 'm', linewidth=1.5, label=r"$\dfrac{\partial^2 a_z}{\partial z^2}g$")

    #ax.plot(np.real(LaplacR11)*fac11, y, 'c', linewidth=1.5, label=r"$\nabla^2 \overline{u'u'}$")

    label_LaplacR33 = r"$\nabla^2 \overline{w'w'}$ ( scaling = %s )" % str('{:.2e}'.format(fac33))
    ax.plot(np.real(LaplacR33)*fac33, y, 'm', linewidth=1.5, label=label_LaplacR33)
    #ax.plot(np.real(LaplacTotal)*factot, y, 'c-.', linewidth=1.5, label=r"Total Laplacian")

    label_azdpbardz = r"$a_z \dfrac{\partial \bar{p}}{\partial z}$ ( scaling = %s )" % str('{:.2e}'.format(fac_azdpbardz))
    ax.plot(np.real(azdpbardz)*fac_azdpbardz, y, 'c', linewidth=1.5, label=label_azdpbardz)

    
    ax.plot(np.real(az), y, 'm-.', linewidth=1.5, label="$a_z$")
    
    ax.plot(np.real(VelPres), y, 'g--', linewidth=1.5, \
            label=r"$-\left(\overline{u' \dfrac{\partial p'}{\partial x}} + \overline{v' \dfrac{\partial p'}{\partial y}} + \overline{w' \dfrac{\partial p'}{\partial z}}  \right)$")

    plt.xlabel("velocity pressure correlation components", fontsize=18)
    plt.ylabel("z", fontsize=18)
    plt.gcf().subplots_adjust(left=0.17)
    plt.gcf().subplots_adjust(bottom=0.15)
    plt.ylim([ymin, ymax])
    
    plt.legend(loc="upper right")
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15),   # box with lower left corner at (x, y)
              ncol=2, fancybox=True, shadow=True, fontsize=10)
    
    f.show()

    ####################################
    # Plot p'*w' and related quantities
    ####################################

    au = np.divide(compute_inner_prod(reig, ueig), Rho)
    daudx_g = np.multiply(rho_inv, compute_inner_prod(reig, dudx) + compute_inner_prod(ueig, drdx) )*gref

    pu = compute_inner_prod(peig, ueig)
    pw = compute_inner_prod(peig, weig)
    dazdz_g = np.matmul(D1, az)*gref
    d2azdz2_g = np.matmul(D2, az)*gref

    maxp_pw = np.amax(np.abs(pw))

    max_Rhop = np.amax(np.abs(Rhop))
    max_Rhopp = np.amax(np.abs(Rhopp))

    max_daudx_g = np.amax(np.abs(daudx_g))
    max_dazdz_g = np.amax(np.abs(dazdz_g))
    max_d2azdz2_g = np.amax(np.abs(d2azdz2_g))

    fac_Rhop = maxp_pw/max_Rhop
    fac_Rhopp = maxp_pw/max_Rhopp
        
    fac_daudx_g = maxp_pw/max_daudx_g
    fac_dazdz_g = maxp_pw/max_dazdz_g

    fac_dazdz_g_Re = maxp_pw/(max_dazdz_g/Re)
    
    fac_max_d2azdz2_g = maxp_pw/max_d2azdz2_g

    # Other formula for <p'w'> from Daniel paper: consistent-pressure-modeling.pdf
    p_w_prime_other = -np.sqrt(2.*np.real(Et_integrand))*np.matmul(D1, np.real(Et_integrand))
    max_p_w_prime_other = np.max(np.abs(p_w_prime_other))

    fac_p_w_prime_other = maxp_pw/max_p_w_prime_other
    fac_p_w_prime_other_Re = maxp_pw/Re

    
    ptn = plt.gcf().number + 1
    
    f  = plt.figure(ptn)
    ax = plt.subplot(111)
    #ax.plot(np.real(pu), y, 'k', linewidth=1.5, label=r"$\overline{p' u'}$")
    ax.plot(np.real(pw), y, 'r', linewidth=1.5, label=r"$\overline{p' w'}$")

    label_p_w_prime_other = r"$-\sqrt{2k}\dfrac{\partial k}{\partial z}$ ( scaling = %s )" % str('{:.2e}'.format(fac_p_w_prime_other))
    ax.plot(np.real(p_w_prime_other)*fac_p_w_prime_other, y, 'k--', linewidth=1.5, label=label_p_w_prime_other)

    label_dazdz_g_Re = r"$g \dfrac{\partial a_z}{\partial z}/Re$ ( scaling = %s )" % str('{:.2e}'.format(fac_dazdz_g_Re))
    ax.plot(np.real(dazdz_g)/Re*fac_dazdz_g_Re, y, 'c--', linewidth=1.5, label=label_dazdz_g_Re)
    
    #label_dazdz_g = r"$g \dfrac{\partial a_z}{\partial z}$ ( scaling = %s )" % str('{:.2e}'.format(fac_dazdz_g))
    #ax.plot(np.real(dazdz_g)*fac_dazdz_g, y, 'b', linewidth=1.5, label=label_dazdz_g)

    #label_Rhop = r"$\dfrac{\partial \bar{\rho}}{\partial z}$ ( scaling = %s )" % str('{:.2e}'.format(fac_Rhop))
    #ax.plot(Rhop*fac_Rhop, y, 'k', linewidth=1.5, label=label_Rhop)

    #label_Rhopp = r"$\dfrac{\partial^2 \bar{\rho}}{\partial z^2}$ ( scaling = %s )" % str('{:.2e}'.format(fac_Rhopp))
    #ax.plot(Rhopp*fac_Rhopp, y, 'g', linewidth=1.5, label=label_Rhopp)

    #label_daudx_g = "$g \dfrac{\partial a_u}{\partial x}$ ( scaling = %s )" % str('{:.2e}'.format(fac_daudx_g))
    #ax.plot(np.real(d2azdz2_g)*fac_max_d2azdz2_g, y, 'g', linewidth=1.5, label=label_daudx_g)

    plt.xlabel("velocity pressure correlation --> more components", fontsize=18)
    plt.ylabel("z", fontsize=18)
    plt.gcf().subplots_adjust(left=0.17)
    plt.gcf().subplots_adjust(bottom=0.15)
    #plt.ylim([-0.05, 0.05])
    plt.ylim([ymin, ymax])
    
    plt.legend(loc="upper right")
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15),   # box with lower left corner at (x, y)
              ncol=2, fancybox=True, shadow=True, fontsize=10)
    
    f.show()


    
    #####################################################################
    # Compute Energy balance for all equations independently
    #####################################################################

    DoThis = False

    if (DoThis):
        
        # x-momentum
        LHS_xmom_tmp = 0.5*compute_inner_prod(ueig, ueig)*2*omega_i 
        LHS_xmom     = np.multiply(LHS_xmom_tmp, Rho)
        
        visc_grad1_xmom  = compute_inner_prod(ueig, Du) 
        visc_grad2_xmom  = compute_inner_prod(ueig, ia*weig) 
        
        visc_grad_t_xmom = np.multiply(Mup, visc_grad1_xmom + visc_grad2_xmom)
        
        visc_xmom = np.multiply(Mu, visc1)
        
        RHS_xmom  = vp1 + visc_xmom + visc_grad_t_xmom
        
        
        # y-momentum
        LHS_ymom_tmp = 0.5*compute_inner_prod(veig, veig)*2*omega_i 
        LHS_ymom     = np.multiply(LHS_ymom_tmp, Rho)
        
        visc_grad1_ymom  = compute_inner_prod(veig, Dv) 
        visc_grad2_ymom  = compute_inner_prod(veig, ib*weig) 
        
        visc_grad_t_ymom = np.multiply(Mup, visc_grad1_ymom + visc_grad2_ymom)
        
        visc_ymom = np.multiply(Mu, visc2)
        
        RHS_ymom  = vp2 + visc_ymom + visc_grad_t_ymom
        
        # z-momentum
        LHS_zmom_tmp = 0.5*compute_inner_prod(weig, weig)*2*omega_i 
        LHS_zmom     = np.multiply(LHS_zmom_tmp, Rho)
        
        visc_grad1_zmom  = compute_inner_prod(weig, Dw) 
        visc_grad2_zmom  = compute_inner_prod(weig, Dw) 
        
        visc_grad_t_zmom = np.multiply(Mup, visc_grad1_zmom + visc_grad2_zmom)
        
        visc_zmom = np.multiply(Mu, visc3)
        
        grav_zmom = -compute_inner_prod(reig, weig)*9.81 
        
        RHS_zmom  = vp3 + visc_zmom + visc_grad_t_zmom + grav_zmom

        
    plot_each_eq = False

    if (plot_each_eq):
    
        ptn = ptn + 1
        
        # X-MOMENTUM
        f = plt.figure(ptn)
        plt.plot(LHS_xmom, y, 'b', linewidth=1.5, label=r"LHS (x-mom)")
        plt.plot(RHS_xmom, y, 'm--', linewidth=1.5, label=r"RHS (x-mom)")
        plt.xlabel("Energy balance R-T X-MOMENTUM", fontsize=18)
        plt.ylabel("z", fontsize=18)
        plt.gcf().subplots_adjust(left=0.17)
        plt.gcf().subplots_adjust(bottom=0.15)
        #plt.ylim([-0.03, 0.03])
        f.show()
        
        plt.legend(loc="upper center")
        
        ptn = ptn + 1
        
        # Y-MOMENTUM
        f = plt.figure(ptn)
        plt.plot(LHS_ymom, y, 'b', linewidth=1.5, label=r"LHS (y-mom)")
        plt.plot(RHS_ymom, y, 'm--', linewidth=1.5, label=r"RHS (y-mom)")
        plt.xlabel("Energy balance R-T Y-MOMENTUM", fontsize=18)
        plt.ylabel("z", fontsize=18)
        plt.gcf().subplots_adjust(left=0.17)
        plt.gcf().subplots_adjust(bottom=0.15)
        #plt.ylim([-0.03, 0.03])
        f.show()
        
        plt.legend(loc="upper center")
        
        ptn = ptn + 1
        
        # Z-MOMENTUM
        f = plt.figure(ptn)
        plt.plot(LHS_zmom, y, 'b', linewidth=1.5, label=r"LHS (z-mom)")
        plt.plot(RHS_zmom, y, 'm--', linewidth=1.5, label=r"RHS (z-mom)")
        plt.xlabel("Energy balance R-T Z-MOMENTUM", fontsize=18)
        plt.ylabel("z", fontsize=18)
        plt.gcf().subplots_adjust(left=0.17)
        plt.gcf().subplots_adjust(bottom=0.15)
        #plt.ylim([-0.03, 0.03])
        f.show()
        
        plt.legend(loc="upper center")

    #print("")
    #input("R-T: LHS/RHS Kinetic Energy Equation")
    
    # # a_i (see Boussinesq paper by D. Israel page 17)
    # rhoxw = compute_inner_prod(reig, weig)
    # #a_i  = -np.divide(rhoxw, Rho) ==> bug fix ==> no leading minus sign

    # a_i   = -np.divide(rhoxw, Rho)
    
    # a_1   = -np.divide(rhoxw, Rho)
    # a_2   = -np.divide(rhoxw, Rho)
    # a_3   = -np.divide(rhoxw, Rho)

    # #print("a_i = ", a_i)

    # # Compute b (see Boussinesq paper by D. Israel Eq. 22b)
    # rhoxrho = compute_inner_prod(reig, reig)
    # #b_      = -np.divide(rhoxrho, np.multiply(Rho, Rho) ) ==> bug fixed on December 8 2022 ==> no leading minus sign
    # b_      = np.divide(rhoxrho, np.multiply(Rho, Rho) )

    # Rho2    = np.multiply(Rho, Rho)
    # #print("b_ = ", b_)

    # #print("1/Rho2 = ", 1/Rho2)
    
    # # compute rho'*dp'/dz
    # rhodpdz   = compute_inner_prod(reig, np.matmul(D1, peig))
    # Pres_corr = -np.multiply(1/Rho2, rhodpdz)

    # Cap             = 0.1 
    # Pres_corr_model = Cap*np.multiply( np.divide(b_, Rho), -Rho*9.81 )

    # # Dissipation ----- Other way 333
    # term1_other333 = compute_inner_prod(1j*alpha*ueig, 1j*alpha*ueig)
    # term2_other333 = compute_inner_prod(np.matmul(D1, ueig), np.matmul(D1, ueig))
    # term3_other333 = compute_inner_prod(1j*alpha*veig, 1j*alpha*veig)
    # term4_other333 = compute_inner_prod(np.matmul(D1, veig), np.matmul(D1, veig))

    # # I do not have the Reynolds number here ==> use nu instead
    # nu = np.divide(Mu, Rho)
    # Dissipation_integrand_other333 = -np.multiply(nu, term1_other333 + term2_other333 + term3_other333 + term4_other333 )

    # C1a = 2.8
    # Pres_corr_model_full = Cap*np.multiply( np.divide(b_, Rho), -Rho*9.81 ) + C1a*np.multiply( np.divide(np.real(Dissipation_integrand_other333), np.real(Et_integrand)), a_i)

    # ptn = ptn + 1

    # print("np.amax(np.imag(a_i)) = ", np.amax(np.imag(a_i)))
    # print("np.amax(np.imag(b_)) = ", np.amax(np.imag(b_)))
    # print("np.amax(np.imag(Dissipation_integrand_other333)) = ", np.amax(np.imag(Dissipation_integrand_other333)))
    
    # f = plt.figure(ptn)
    # plt.plot(np.real(a_i), y, 'k', linewidth=1.5, label=r"$a_i$")
    # plt.plot(np.real(b_), y, 'r', linewidth=1.5, label=r"$b$")
    # plt.plot(np.real(Dissipation_integrand_other333), y, 'b', linewidth=1.5, label=r"$\epsilon$")
    # #plt.plot(Rho2, y, 'g', linewidth=1.5, label=r"$\bar{\rho}^2$")
    
    # plt.xlabel("a, b, epsilon", fontsize=18)
    # plt.ylabel("z", fontsize=18)
    # plt.gcf().subplots_adjust(left=0.17)
    # plt.gcf().subplots_adjust(bottom=0.15)
    # plt.ylim([-0.03, 0.03])
    # f.show()

    # plt.legend(loc="upper center")

    # ptn = ptn + 1

    # f = plt.figure(ptn)
    # plt.plot(np.real(Pres_corr), y, 'k', linewidth=1.5, label=r"Pressure correlation (exact)")
    # plt.plot(np.real(Pres_corr_model), y, 'r', linewidth=1.5, label=r"Pressure correlation (model)")
    # #plt.plot(np.real(Pres_corr_model_full), y, 'r', linewidth=1.5, label=r"Pressure correlation (full model)")
    
    # plt.xlabel("Pressure correlation", fontsize=18)
    # plt.ylabel("z", fontsize=18)
    # plt.gcf().subplots_adjust(left=0.20)
    # plt.gcf().subplots_adjust(bottom=0.15)
    # plt.ylim([-0.002, 0.002])
    # f.show()

    # plt.legend(loc="upper center")

def check_mass_continuity_satisfied_rayleigh_taylor(ueig, veig, weig, peig, reig, D1, D2, y, alpha, beta, omega_i, bsfl, bsfl_ref, mob, rt_flag):

    #plt.close('all') 
    
    # Get some quantities to compute the balances
    Mu, Mup, Rho, Rhop, Rhopp, rho2, rho3, rho_inv, rho2_inv, rho3_inv = get_baseflow_and_derivatives(bsfl, mob)
    ia, ib, ia2, ib2, iab, Du, Dv, Dw, Dr, D2u, D2v, D2w, D2r = get_eigenfunctions_quantities(alpha, beta, D1, D2, ueig, veig, weig, reig)
    
    Sc = bsfl_ref.Sc
    Re = bsfl_ref.Re

    Lref = bsfl_ref.Lref
    Uref = bsfl_ref.Uref
    rhoref = bsfl_ref.rhoref

    #
    # "Balance" Equations for Continuity and Mass ==> I multiply through by rho'
    #
    
    if   ( mob.boussinesq == 1 ):
        LHS_conti = compute_inner_prod(reig, ia*ueig) + compute_inner_prod(reig, ib*veig) + compute_inner_prod(reig, Dw)
        RHS_conti = 0.0*LHS_conti

        LHS_mass  = omega_i*compute_inner_prod(reig, reig) + np.multiply(Rhop, compute_inner_prod(reig, weig))
        RHS_mass  = (  compute_inner_prod(reig, ia2*reig) \
                      +compute_inner_prod(reig, ib2*reig) \
                      +compute_inner_prod(reig, D2r) )/(Re*Sc)
        
        #print("omega_i = ", omega_i)
        #print("max ( LHS_mass, RHS_mass ) = ", np.max(np.abs(LHS_mass)), np.max(np.abs(RHS_mass)))

    elif ( mob.boussinesq == -2 ):
        LHS_conti = compute_inner_prod(reig, ia*ueig) + compute_inner_prod(reig, ib*veig) + compute_inner_prod(reig, Dw)
        RHS_conti = 0.0*LHS_conti

        LHS_mass  = omega_i*compute_inner_prod(reig, reig) + np.multiply(Rhop, compute_inner_prod(reig, weig))
        RHS_mass  = 0.0*LHS_mass

        # Nondimensionalize eigenfunctions
        ueig_nd = ueig/Uref
        veig_nd = veig/Uref
        weig_nd = weig/Uref

        peig_nd = peig/(rhoref*Uref**2.)
        reig_nd = reig/rhoref

        norm_s, ueig_nd, veig_nd, weig_nd, peig_nd, reig_nd = normalize_eigenvectors(ueig_nd, veig_nd, weig_nd, peig_nd, reig_nd, rt_flag)

        LHS_mass_nd = omega_i*Lref/Uref*compute_inner_prod(reig_nd, reig_nd) + np.multiply(bsfl.Rhop_nd, compute_inner_prod(reig_nd, weig_nd))
        RHS_mass_nd = 0.0*LHS_mass        

    elif ( mob.boussinesq == -3 or mob.boussinesq == -4 ):
        #print("rho2_inv.shape, Rhop.shape, Rhopp.shape, rho_inv.shape, rho3_inv.shape",\
        #      rho2_inv.shape, Rhop.shape, Rhopp.shape, rho_inv.shape, rho3_inv.shape)

        bsfl1 = 2.*np.multiply(rho2_inv, Rhop)
        bsfl2 = np.multiply(rho2_inv, Rhopp)
        bsfl3 = -rho_inv
        bsfl4 = -2.*np.multiply(rho3_inv, np.multiply(Rhop, Rhop))
        ReSc_continuity   = ( np.multiply(bsfl1, compute_inner_prod(reig, Dr)) \
                              + np.multiply(bsfl2, compute_inner_prod(reig, reig)) \
                              + np.multiply(bsfl3, compute_inner_prod(reig, ia2*reig)) \
                              + np.multiply(bsfl3, compute_inner_prod(reig, ib2*reig)) \
                              + np.multiply(bsfl3, compute_inner_prod(reig, D2r)) \
                              + np.multiply(bsfl4, compute_inner_prod(reig, reig)) )/(Re*Sc)

        bsfl5 = np.multiply(rho2_inv, np.multiply(Rhop, Rhop))
        bsfl6 = -2.*np.multiply(rho_inv, Rhop)
        ReSc_rho_eq_terms = ( compute_inner_prod(reig, ia2*reig) \
                              +compute_inner_prod(reig, ib2*reig) \
                              +compute_inner_prod(reig, D2r) \
                              +np.multiply(bsfl5, compute_inner_prod(reig, reig)) \
                              +np.multiply(bsfl6, compute_inner_prod(reig, Dr)) )/(Re*Sc)

        print("")
        print("Sandoval Equations: Magnitude of RHS terms of mass equation:")
        print("------------------------------------------------------------")
        print("1/(Re*Sc)*( d2rho'dxjdxj ) = ", np.linalg.norm((compute_inner_prod(reig, ia2*reig) + compute_inner_prod(reig, ib2*reig) + compute_inner_prod(reig, D2r))/(Re*Sc)))
        print("1/(Re*Sc)*( drho_bar/dxj drho_bar/dxj *rho'/rho_bar^2 ) = ", np.linalg.norm(np.multiply(bsfl5, compute_inner_prod(reig, reig))/(Re*Sc) ))
        print("1/(Re*Sc)*( -2/rho_bar*drho_bar/dxj*drho'/dxj ) = ", np.linalg.norm(np.multiply(bsfl6, compute_inner_prod(reig, Dr))/(Re*Sc) ))
        
        LHS_conti = compute_inner_prod(reig, ia*ueig) + compute_inner_prod(reig, ib*veig) + compute_inner_prod(reig, Dw)
        RHS_conti = ReSc_continuity

        LHS_mass  = omega_i*compute_inner_prod(reig, reig) + np.multiply(Rhop, compute_inner_prod(reig, weig))
        RHS_mass  = ReSc_rho_eq_terms
    

    PlotIt = 1
    
    if ( ( mob.boussinesq == 1 or mob.boussinesq == -3 or mob.boussinesq == -4 ) and PlotIt == 1 ):

        ptn = plt.gcf().number + 1
        
        f = plt.figure(ptn)
        plt.plot(np.real(LHS_conti), y, 'k', linewidth=1.5, label=r"LHS (continuity)")
        plt.plot(np.real(RHS_conti), y, 'c--', linewidth=1.5, label=r"RHS (continuity)")
        plt.xlabel("LHS vs. RHS for continuity eq.", fontsize=18)
        plt.ylabel("y", fontsize=18)
        plt.gcf().subplots_adjust(left=0.17)
        plt.gcf().subplots_adjust(bottom=0.15)
        #plt.xlim([0.0, 1.e-5])
        #plt.ylim([-0.07, 0.07])
        f.show()
        plt.legend(loc="upper right")


        ptn = ptn + 1
        
        f = plt.figure(ptn)
        plt.plot(np.real(LHS_mass), y, 'k', linewidth=1.5, label=r"LHS (mass)")
        plt.plot(np.real(RHS_mass), y, 'c--', linewidth=1.5, label=r"RHS (mass)")
        plt.xlabel("LHS vs. RHS for mass eq.", fontsize=18)
        plt.ylabel("y", fontsize=18)
        plt.gcf().subplots_adjust(left=0.17)
        plt.gcf().subplots_adjust(bottom=0.15)
        #plt.xlim([0.0, 1.e-5])
        #plt.ylim([-0.07, 0.07])
        f.show()
        plt.legend(loc="upper right")
        

    print("")
    print("Check energy balance: CONTINUITY AND MASS EQUATIONS")
    print("---------------------------------------------------")
    energy_bal_cont = np.amax(np.abs(LHS_conti-RHS_conti))
    energy_bal_mass = np.amax(np.abs(LHS_mass-RHS_mass))

    if ( mob.boussinesq == -2 ):
        energy_bal_mass_nd = np.amax(np.abs(LHS_mass_nd-RHS_mass_nd))
        print("boussinesq == -2")
        print("Energy balance (RT) mass equation (non-dimensional) = ", energy_bal_mass_nd)
        
    print("")

    #input("R-T: LHS/RHS for Continuity and Mass Equations")
    print("")

    #####################################################################
    # MASS EQUATION: from drho'/dt = -w'drho_bar/dz ===> multiply by rho'
    #####################################################################
    
    # Production drhodz
    #Prod_drhodz = -np.multiply( compute_inner_prod(reig, weig), Rhop )

    # time depenedent density
    #Rho2_contrib = 0.5*2*omega_i*compute_inner_prod(reig, reig) 

    # plot_mass_eq = False

    # if (plot_mass_eq):
        
    #     ptn = ptn + 1
        
    #     f = plt.figure(ptn)
    #     plt.plot(Prod_drhodz, y, 'b', linewidth=1.5, label=r"Prod_drhodz")
    #     plt.plot(Rho2_contrib, y, 'm--', linewidth=1.5, label=r"Rho2_contrib")
    #     plt.xlabel("Energy balance R-T", fontsize=18)
    #     plt.ylabel("z", fontsize=18)
    #     plt.gcf().subplots_adjust(left=0.17)
    #     plt.gcf().subplots_adjust(bottom=0.15)
    #     f.show()
        
    #     plt.legend(loc="upper center")






    #LHS_conti = ia*ueig + ib*veig + Dw
    #RHS_conti = ( bsfl1*Dr + bsfl2*reig + bsfl3*( ia2*reig + ib2*reig + D2r ) + bsfl4*reig )/(Re*Sc)
    
    #conti_check = 1j*alpha*ueig + 1j*beta*veig + np.matmul(D1, weig)
    #mass_check  = np.divide(conti_check, conti_check)


    # if (rt_flag):

    #     if   ( mob.boussinesq == -3 ):
            
    #     elif ( mob.boussinesq == 1 ):
    #         pass

    #     elif ( mob.boussinesq == -2 ):
    #         conti_check = 1j*alpha*ueig + 1j*beta*veig + np.matmul(D1, weig)
    #         mass_check  = np.multiply(weig, bsfl.Rhop) - 1j*omega*reig

    # else:

    #     conti_check = 1j*alpha*ueig + np.matmul(D1, veig) + 1j*beta*weig

    # conti_check_max = np.max(np.abs(conti_check))
    # print("Continuity residual = ", conti_check_max)
    
    # if (rt_flag):
    #     mass_check_max = np.max(np.abs(mass_check))
    #     print("Extra R-T check     = ", mass_check_max)



    
def get_baseflow_and_derivatives(bsfl, mob):

    if   ( mob.boussinesq == 1 or mob.boussinesq == -3 or mob.boussinesq == -4 ):
        Rho   = bsfl.Rho_nd
        Rhop  = bsfl.Rhop_nd
        Rhopp = bsfl.Rhopp_nd

        rho2    = np.multiply(Rho, Rho)
        rho3    = np.multiply(rho2, Rho)

        rho_inv  = 1/Rho
        rho2_inv = 1/rho2
        rho3_inv = 1/rho3

        Mu       = bsfl.Mu_nd
        Mup      = bsfl.Mup_nd

    elif ( mob.boussinesq == -2 or mob.boussinesq == 10 ):
        Rho   = bsfl.Rho
        Rhop  = bsfl.Rhop
        Rhopp = bsfl.Rhopp
        
        Mu   = bsfl.Mu
        Mup  = bsfl.Mup

        rho2    = np.multiply(Rho, Rho)
        rho3    = np.multiply(rho2, Rho)

        rho_inv  = 1/Rho
        rho2_inv = 1/rho2
        rho3_inv = 1/rho3
        
        #sys.exit("Check: might need other baseflow quantities!!")
    else:
        sys.exit("Flag error")

    return Mu, Mup, Rho, Rhop, Rhopp, rho2, rho3, rho_inv, rho2_inv, rho3_inv







def get_eigenfunctions_quantities(alpha, beta, D1, D2, ueig, veig, weig, reig):

    # Viscous (proportional to mu_baseflow)
    ia  = ( 1j*alpha )
    ib  = ( 1j*beta )
    
    ia2 = ( 1j*alpha )**2.
    ib2 = ( 1j*beta )**2.

    iab = ( 1j*alpha )*( 1j*beta )

    Du  = np.matmul(D1, ueig)
    Dv  = np.matmul(D1, veig)
    Dw  = np.matmul(D1, weig)
    Dr  = np.matmul(D1, reig)

    D2u = np.matmul(D2, ueig)
    D2v = np.matmul(D2, veig)
    D2w = np.matmul(D2, weig)
    D2r = np.matmul(D2, reig)

    return ia, ib, ia2, ib2, iab, Du, Dv, Dw, Dr, D2u, D2v, D2w, D2r



def normalize_eigenvectors(ueig_vec, veig_vec, weig_vec, peig_vec, reig_vec, rt_flag):

    multiply_by_1j = 0
    normalize = 1
        
    max_array = np.zeros(5, dp)
    idx_max_a = np.zeros(5, i4)

    max_array[0] = np.max(np.abs(ueig_vec))
    idx_max_a[0] = np.argmax(np.abs(ueig_vec))
    
    max_array[1] = np.max(np.abs(veig_vec))
    idx_max_a[1] = np.argmax(np.abs(veig_vec))
    
    max_array[2] = np.max(np.abs(weig_vec))
    idx_max_a[2] = np.argmax(np.abs(weig_vec))
    
    max_array[3] = np.max(np.abs(peig_vec))
    idx_max_a[3] = np.argmax(np.abs(peig_vec))
        
    max_array[4] = np.max(np.abs(reig_vec))
    idx_max_a[4] = np.argmax(np.abs(reig_vec))

    # Global max
    idx_array = np.argsort(max_array)
    idx_max   = idx_array[-1] # because it sorts in ascending order

    # print("max_array[0] = ", max_array[0])
    # print("max_array[1] = ", max_array[1])
    # print("max_array[2] = ", max_array[2])
    # print("max_array[3] = ", max_array[3])
    # print("max_array[4] = ", max_array[4])

    # print("")
    # print("max_array[idx_max] = ", max_array[idx_max])

    global_max = np.max(max_array)
    print("")
    print("global_max (absolute value) = ", global_max)

    global_max_idx = idx_max_a[idx_max]
    print("global_max_idx = ", global_max_idx)
    
    #print("ueig_vec[global_max_idx] = ", ueig_vec[global_max_idx])
    #print("veig_vec[global_max_idx] = ", veig_vec[global_max_idx])
    #print("weig_vec[global_max_idx] = ", weig_vec[global_max_idx])
    #print("reig_vec[global_max_idx] = ", reig_vec[global_max_idx])
    #print("peig_vec[global_max_idx] = ", peig_vec[global_max_idx])

    print("")
    if   (idx_max==0):
        print("Scaling eigenfunctions with max. of u-vel")
        norm_s      = ueig_vec[global_max_idx] #np.max(np.abs(ueig_vec))
    elif (idx_max==1):
        print("Scaling eigenfunctions with max. of v-vel")
        norm_s      = veig_vec[global_max_idx] #np.max(np.abs(veig_vec))
    elif (idx_max==2):
        print("Scaling eigenfunctions with max. of w-vel")
        norm_s      = weig_vec[global_max_idx] #np.max(np.abs(weig_vec))
    elif (idx_max==3):
        print("Scaling eigenfunctions with max. of pressure")
        norm_s      = peig_vec[global_max_idx] #np.max(np.abs(peig_vec))
    elif (idx_max==4):
        print("Scaling eigenfunctions with max. of density")
        norm_s      = reig_vec[global_max_idx] #np.max(np.abs(reig_vec))

    print("norm_s = ", norm_s)

    #print("max_array = ", max_array)
    #print("idx_max = ", idx_max)
    #input("debug")
    
    # normalizing u and v by max(abs(u))
    #norm_s      = np.max(np.abs(ueig_vec))
    #norm_s_conj = np.max(np.abs(ueig_conj_vec))

    if (normalize==1):
        #print("ueig_vec = ", ueig_vec)
        ueig_vec = ueig_vec/norm_s
        veig_vec = veig_vec/norm_s
        weig_vec = weig_vec/norm_s
        peig_vec = peig_vec/norm_s

        if (rt_flag):
            reig_vec    = reig_vec/norm_s

    if (multiply_by_1j==1):
        print("")
        print("Multiplying eigenfunctions by 1j in normalize_eigenvectors")
        print("")
        ueig_vec = ueig_vec*1j
        veig_vec = veig_vec*1j
        weig_vec = weig_vec*1j
        reig_vec = reig_vec*1j
        peig_vec = peig_vec*1j
        

    return norm_s, ueig_vec, veig_vec, weig_vec, peig_vec, reig_vec






def write_energy_balance_terms(npts, y, t1, t2, t3, filename):
    
    work_dir      = "./"
    save_path     = work_dir #+ '/' + folder1
    completeName  = os.path.join(save_path, filename)
    fileoutFinal  = open(completeName,'w')
    
    zname1D       = 'ZONE T="1-D Zone", I = ' + str(npts) + '\n'
    #fileoutFinal.write('TITLE     = "1d energy balance data"\n')
    #fileoutFinal.write('VARIABLES = "Y" "VelPres" "Visc" "Prod"\n')
    #fileoutFinal.write(zname1D)
    
    for i in range(0, npts):
        fileoutFinal.write("%25.15e %25.15e %25.15e %25.15e \n" % ( y[i], t1[i], t2[i], t3[i] ) )

    fileoutFinal.close()


def build_total_flow_field(reig, Rho, alpha, beta, omega, z):

    sys.exit("stop need to check this!!!!!!!!")
    
    x = 0.0
    y = 0.0
    # time
    t = 0.0

    ny = len(reig)

    amp_r    = np.abs(reig)
    phase_r  = np.arctan2(reig.imag, reig.real)
    #print("phase_r = ", phase_r)

    # Compute the maximum of baseflow and eigenfunction amplitude
    max_bsfl = Rho[-1]-Rho[0] #np.max(Rho)
    #max_eigf = np.max(amp_r)

    shift_val = Rho[0]

    #print("shift_val = ", shift_val)

    #print("max_bsfl = ", max_bsfl)
    #print("")

    # Set ref val used to scale the disturbance flow field
    ref_val  = 0.10*max_bsfl

    #print("ref_val = ", ref_val)
    #amp_r    = amp_r/max_eigf*ref_val

    #print("max(amp_r) = ", np.max(amp_r))

    for k in range(0, ny):
        cmplx_fct_loc = np.exp( 1j*( alpha*x + beta*y - omega*t + phase_r[k] )  )
        rho_prime_loc = np.real( amp_r[k]*cmplx_fct_loc )

        #if ( np.abs(rho_prime_loc) > 1.e-3):
            #print("cmplx_fct_loc, rho_prime_loc = ", cmplx_fct_loc, rho_prime_loc)
        
    #input("Check this data 1111111")

    cmplx_fct = np.exp( 1j*( alpha*x + beta*y - omega*t + phase_r )  )
    rho_prime = np.real( amp_r*cmplx_fct )

    #print("rho_prime = ", rho_prime)

    #print("np.max(rho_prime) = ", np.max(rho_prime))

    rho_prime = rho_prime/np.max(np.abs(rho_prime))*ref_val

    print("")
    print("Max. rho_prime = ", np.amax(rho_prime))
    print("Min. rho_prime = ", np.amin(rho_prime))

    dist_perc = np.max(np.abs(rho_prime))/max_bsfl*100
    print("Max of disturbance density is %15.10f %% of max. baseflow density" % dist_perc)

    #print("cmplx_fct = ", cmplx_fct)
    #print("max(rho_prime) = ", np.max(rho_prime))
    
    rho_total = Rho + rho_prime

    ptn = plt.gcf().number + 1

    label_dist = "Disturbance density (shifted, %s %%)" % str('{:.2f}'.format(dist_perc)) 
    
    f = plt.figure(ptn)
    plt.plot(Rho, z, 'k', linewidth=1.5, label=r"Baseflow density")
    plt.plot(rho_prime+shift_val, z, 'r', linewidth=1.5, label=label_dist)
    plt.plot(rho_total, z, 'g--', linewidth=1.5, label=r"Total density")
    #
    plt.xlabel("Density", fontsize=18)
    plt.ylabel("z", fontsize=18)
    
    plt.gcf().subplots_adjust(left=0.18)
    plt.gcf().subplots_adjust(bottom=0.15)
    #plt.xlim([0.45, 1.7])
    #plt.ylim([-0.00025, 0.00025])
    
    plt.legend(loc="upper center", fontsize = 11)
    f.show()
        
    input("R-T: Density plots")


def write_2d_eigenfunctions(weig, reig, alpha, beta, omega, z, bsfl, mob):
    
    nx = 501
    nz = len(z)

    amp_r    = np.abs(reig)
    phase_r  = np.arctan2(reig.imag, reig.real)

    amp_w    = np.abs(weig)
    phase_w  = np.arctan2(weig.imag, weig.real)

    t    = 0.0
    xmin = 0.0
    xmax = 0.05

    x = np.linspace(xmin, xmax, nx)
    y = 0.0

    # Get baseflow
    Mu, Mup, Rho, Rhop, Rhopp, rho2, rho3, rho_inv, rho2_inv, rho3_inv = get_baseflow_and_derivatives(bsfl, mob)

    # Compute the maximum of baseflow and eigenfunction amplitude
    max_bsfl = Rho[-1]-Rho[0] #np.max(Rho)

    # Set ref val used to scale the disturbance flow field
    ref_val  = 0.50*max_bsfl
    
    # ii=21
    # for k in range(0, nz):
    #     cmplx_fct_loc = np.exp( 1j*( alpha*x[ii] + beta*y[k] - omega*t + phase_r[k] )  )
    #     rho_prime_loc = np.real( amp_r[k]*cmplx_fct_loc )

    #     if ( np.abs(rho_prime_loc) > 1.e-3):
    #         print("cmplx_fct_loc, rho_prime_loc = ", cmplx_fct_loc, rho_prime_loc)
        
    # input("Check this data")

    cmplx_fct_w = np.zeros((nx,nz), dpc)
    cmplx_fct_r = np.zeros((nx,nz), dpc)

    w_prime = np.zeros((nx,nz), dp)
    r_prime = np.zeros((nx,nz), dp)
    
    for k in range(0, nz):
        for i in range(0, nx):

            cmplx_fct_r[i,k] = np.exp( 1j*( alpha*x[i] + beta*y - omega*t + phase_r[k] )  )
            r_prime[i,k] = np.real( amp_r[k]*cmplx_fct_r[i,k] )
            
            cmplx_fct_w[i,k] = np.exp( 1j*( alpha*x[i] + beta*y - omega*t + phase_w[k] )  )
            w_prime[i,k] = np.real( amp_w[k]*cmplx_fct_w[i,k] )


    scale_fac = 1/np.amax(np.abs(r_prime))*ref_val
    
    r_prime = r_prime*scale_fac
    w_prime = w_prime*scale_fac

    print("")
    print("Max (r_prime) = ", np.amax(np.abs(r_prime)))
            
    filename      = 'Eigenfunction_2d_wr.dat'
    filename1     = 'Baseflow_density.dat'

    work_dir      = "./"
    save_path     = work_dir #+ '/' + folder1
    
    completeName  = os.path.join(save_path, filename)
    fileoutFinal  = open(completeName,'w')
    
    zname2D       = 'ZONE T="2-D Zone", I = ' + str(nx) + ', J = ' + str(nz) + '\n'
    fileoutFinal.write('TITLE     = "2D DATA"\n')
    fileoutFinal.write('VARIABLES = "X" "Z" "dens dist" "w-vel dist"\n')
    fileoutFinal.write(zname2D)
    
    for k in range(0, nz):
        for i in range(0, nx):
            #cmplx_fct_loc = np.exp( 1j*( alpha*x[i] + beta*y[k] - omega*t + phase_r[k] )  )
            #rho_prime_loc = np.real( amp_r[k]*cmplx_fct_loc )

            #rho_prime_loc = rho_prime_loc/np.max(np.abs(rho_prime))*ref_val

            #cmplx_fct_w_loc = np.exp( 1j*( alpha*x[i] + beta*y[k] - omega*t + phase_w[k] )  )
            #w_prime_loc = np.real( amp_w[k]*cmplx_fct_w_loc )

            #print("rho_prime_loc = ", rho_prime_loc)
            fileoutFinal.write("%25.15e %25.15e %25.15e %25.15e \n" % ( x[i], z[k], w_prime[i,k], r_prime[i,k] ) )

    fileoutFinal.close()

    # Now write baseflow to file
    completeName1  = os.path.join(save_path, filename1)
    fileoutFinal1  = open(completeName1,'w')

    zname2D       = 'ZONE T="2-D Zone", I = ' + str(nx) + ', J = ' + str(nz) + '\n'
    fileoutFinal1.write('TITLE     = "2D DATA"\n')
    fileoutFinal1.write('VARIABLES = "X" "Z" "dens baseflow"\n')
    fileoutFinal1.write(zname2D)
    
    for k in range(0, nz):
        for i in range(0, nx):
            
            fileoutFinal1.write("%25.15e %25.15e %25.15e \n" % ( x[i], z[k], Rho[k] ) )

    fileoutFinal1.close()

def check_baseflow_disturbance_flow_divergence_boussinesq(ueig, veig, weig, peig, reig, alpha, beta, bsfl, bsfl_ref, map, mob):

    y = map.y
    ny = len(y)
    
    D1 = map.D1

    Sc = bsfl_ref.Sc
    Re = bsfl_ref.Re

    # velocity induced for Boussinesq LST
    wvel_base_lst_steady_boussi = 1/(Re*Sc)*np.divide(Rhopp, Rhop)
    wvel_base_lst_steady_exact_boussi = 1/(Re*Sc)*(-2.*y/bsfl.delta**2.)

    # This does not work with spectral matrix because vel_boussi_lst_steady has some nan's
    #dwdz_boussi_lst = np.matmul(D1, vel_boussi_lst_steady)
    dwdz_boussi_lst_exact = 1/(Re*Sc)*(-2./bsfl.delta**2.)*np.ones(ny)
    dwdz_boussi_lst = np.zeros(ny, dp)
    dwdz_boussi_lst[0] = ( vel_boussi_lst_steady[1]-vel_boussi_lst_steady[0] )/(y[1]-y[0])
    dwdz_boussi_lst[-1] = ( vel_boussi_lst_steady[-1]-vel_boussi_lst_steady[-2] )/(y[-1]-y[-2])
    
    for i in range(1, ny-1):
        dwdz_boussi_lst[i] = ( vel_boussi_lst_steady[i+1]-vel_boussi_lst_steady[i-1] )/(y[i+1]-y[i-1])

    array_nan_check = np.isnan(vel_boussi_lst)

    i=0
    while ( array_nan_check[i] == True ):
        i = i+1
        ibeg = i
        #print("i, array_nan_check[i] = ", i, array_nan_check[i])

    i=-1
    while ( array_nan_check[i] == True ):
        i = i-1
        ifin = i
        #print("i, array_nan_check[i] = ", i, array_nan_check[i])

    #print("ibeg = ", ibeg)
    #print("ifin = ", ifin)

    #print("array_nan_check[ibeg:ifin+1]", array_nan_check[ibeg:ifin+1])


    
def check_baseflow_disturbance_flow_divergence_sandoval(ueig, veig, weig, peig, reig, alpha, beta, bsfl, bsfl_ref, map, mob):

    y = map.y
    ny = len(y)
    
    D1 = map.D1

    Sc = bsfl_ref.Sc
    Re = bsfl_ref.Re

    ################
    # 1) BASEFLOW
    ################
    
    Mu, Mup, Rho, Rhop, Rhopp, rho2, rho3, rho_inv, rho2_inv, rho3_inv = get_baseflow_and_derivatives(bsfl, mob)

    drhodz2 = np.multiply(Rhop, Rhop)

    term1 = 1/(Re*Sc)*( np.multiply(rho2_inv, drhodz2) )
    term2 = -1/(Re*Sc)*np.multiply(rho_inv, Rhopp)

    # Baseflow divergence for sandoval equations
    div = term1 + term2

    wvel_base_sando = trapezoid_integration_cum(div, y)
    wvel_base_sando_exact = -1/(Re*Sc)*np.multiply(rho_inv, Rhop)

    dwdz_sando = np.matmul(D1, wvel_base_sando_exact)

    RhopRhop = np.multiply(Rhop, Rhop)
        
    wvel_base_lst_steady_sando = -np.divide( np.multiply(Rho, div), Rhop)
    #wvel_base_lst_steady_sando22 = np.divide( -np.multiply(rho_inv, RhopRhop) + Rhopp, Rhop*Re*Sc )
    wvel_base_lst_steady_exact_sando = 1/(Re*Sc)*( -np.multiply(rho_inv, Rhop) -2.*y/bsfl.delta**2.)

    drhodz_dist = np.matmul(D1, reig)
    wvel_base_drhodz_dist = np.multiply(wvel_base_sando_exact, drhodz_dist)
    
    wvel_dist_drhodz_base = np.multiply(weig, Rhop)

    ######################
    # 2) DISTURBANCE FLOW
    ######################

    #div_dist = 1j*alpha*ueig + 1j*beta*veig + np.matmul(D1, weig)
    #print("")
    #print("Max. value of divergence of disturbance field, max(div_dist) = ", np.amax(np.abs(div_dist)))
    #print("")

    
    #######################################
    # PLOTS
    #######################################
    
    ymin = -0.001
    ymax = -ymin

    ptn = plt.gcf().number + 1
    
    f  = plt.figure(ptn)
    ax = plt.subplot(111)
    ax.plot(term1, y, 'k', linewidth=1.5, label=r"$\dfrac{1}{Re Sc} \dfrac{1}{\rho^2} \dfrac{\partial \rho}{\partial z} \dfrac{\partial \rho}{\partial z}$")
    ax.plot(term2, y, 'r', linewidth=1.5, label=r"$-\dfrac{1}{Re Sc} \dfrac{1}{\rho} \dfrac{\partial^2 \rho}{\partial z^2}$")
    ax.plot(div, y, 'b', linewidth=1.5, label=r"$\dfrac{\partial \bar{u}}{\partial x} + \dfrac{\partial \bar{v}}{\partial y} + \dfrac{\partial \bar{w}}{\partial z}$")
    ax.plot(dwdz_sando, y, 'm--', linewidth=1.5, label=r"$\dfrac{\partial w}{\partial z}$ (numerical after computing w)")
    #ax.plot(dwdz_boussi_lst[ibeg:ifin+1], y[ibeg:ifin+1], 'g', linewidth=1.5, label=r"dwdz (Boussinesq LST)")
    #ax.plot(dwdz_boussi_lst_exact, y, 'k--', linewidth=1.5, label=r"dwdz (Boussinesq LST, exact)")
        
    plt.xlabel(r"baseflow divergence", fontsize=14)
    plt.ylabel('y', fontsize=16)
    plt.gcf().subplots_adjust(left=0.16)
    plt.gcf().subplots_adjust(bottom=0.15)
    plt.legend(loc="upper right")
    plt.ylim([ymin, ymax])
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15),   # box with lower left corner at (x, y)
              ncol=2, fancybox=True, shadow=True, fontsize=10)
    
    f.show()

    

    ptn = ptn + 1
    
    f = plt.figure(ptn)
    ax = plt.subplot(111)
    ax.plot(wvel_base_sando, y, 'b', linewidth=1.5, label=r"w-vel (numerical)")
    ax.plot(wvel_base_sando_exact, y, 'g--', linewidth=1.5, label=r"w-vel (exact)")
    ax.plot(wvel_base_lst_steady_sando, y, 'm', linewidth=1.5, label=r"w-vel (lst, steady baseflow)")
    #ax.plot(wvel_base_lst_steady_sando22, y, 'y', linewidth=1.5, label=r"w-vel (lst, steady baseflow TODAY)")
    ax.plot(wvel_base_lst_steady_exact_sando, y, 'c--', linewidth=1.5, label=r"w-vel (lst, steady baseflow, exact)")
    
    #ax.plot(wvel_base_lst_steady_boussi[ibeg:ifin+1], y[ibeg:ifin+1], 'k', linewidth=1.5, label=r"w-vel (Boussinesq LST)")
    #ax.plot(wvel_base_lst_steady_exact_boussi, y, 'r--', linewidth=1.5, label=r"w-vel (Boussinesq LST, exact)")
        
    plt.xlabel(r"velocity induced by density gradients", fontsize=14)
    plt.ylabel('y', fontsize=16)
    plt.gcf().subplots_adjust(left=0.16)
    plt.gcf().subplots_adjust(bottom=0.15)
    plt.legend(loc="upper right")
    plt.ylim([ymin, ymax])
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15),   # box with lower left corner at (x, y)
              ncol=2, fancybox=True, shadow=True, fontsize=10)

    f.show()



    ptn = ptn + 1
    
    f = plt.figure(ptn)
    ax = plt.subplot(111)
    ax.plot(np.real(wvel_base_drhodz_dist), y, 'k', linewidth=1.5, label=r"$\bar{w} \dfrac{\partial \hat{\rho}}{\partial z}$ (real)")
    ax.plot(np.real(wvel_dist_drhodz_base), y, 'c--', linewidth=1.5, label=r"$\hat{w} \dfrac{\partial \bar{\rho}}{\partial z}$ (real)")

    ax.plot(np.imag(wvel_base_drhodz_dist), y, 'b', linewidth=1.5, label=r"$\bar{w} \dfrac{\partial \hat{\rho}}{\partial z}$ (imag)")
    ax.plot(np.imag(wvel_dist_drhodz_base), y, 'r--', linewidth=1.5, label=r"$\hat{w} \dfrac{\partial \bar{\rho}}{\partial z}$ (imag)")

    plt.xlabel(r"check validity of not keeping $\bar{w}$", fontsize=14)
    plt.ylabel('y', fontsize=16)
    plt.gcf().subplots_adjust(left=0.16)
    plt.gcf().subplots_adjust(bottom=0.15)
    plt.legend(loc="upper right")
    plt.ylim([ymin, ymax])
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15),   # box with lower left corner at (x, y)
              ncol=2, fancybox=True, shadow=True, fontsize=10)

    f.show()



    
    input("plot baseflow divergence and velocity")



def compute_baseflow_dissipation(dudx, dudy, dudz, dvdx, dvdy, dvdz, dwdx, dwdy, dwdz, y, bsfl_ref, mob):

    # If non-dimensional equations, visc = 1/Re

    # The vertical direction is z

    # Baseflow terms

    testing = 1

    if testing == 1:
        print("")
        print("---------------> testing in compute_baseflow_dissipation")
        print("")

    if (mob.boussinesq == -2 or mob.boussinesq == 10):
        visc = bsfl_ref.muref
    else:
        visc = 1/bsfl_ref.Re
    
    if (testing == 1):

        nz = len(dudx)

        dudx = np.random.rand(nz)
        dudy = np.random.rand(nz)
        dudz = np.random.rand(nz)
        
        dvdx = np.random.rand(nz)
        dvdy = np.random.rand(nz)
        dvdz = np.random.rand(nz)
        
        dwdx = np.random.rand(nz)
        dwdy = np.random.rand(nz)
        dwdz = np.random.rand(nz)


    sub1 = dudy+dvdx
    sub2 = dvdz+dwdy
    sub3 = dudz+dwdx

    div  = dudx+dvdy+dwdz

    term1 = 2.*( np.multiply(dudx, dudx) + np.multiply(dvdy, dvdy) + np.multiply(dwdz, dwdz) )
    term2 = -2./3.*np.multiply(div, div)
    term3 = np.multiply(sub1, sub1) + np.multiply(sub2, sub2) + np.multiply(sub3, sub3)

    epsilon_check = visc*( term1 + term2 + term3 )

    ########################################
    # Various components of the dissipation
    ########################################

    # 1) visc*(du_i/dx_j)*(du_i/dx_j)
    epsilon_tilde_base = np.multiply(dudx, dudx) + np.multiply(dudy, dudy) + np.multiply(dudz, dudz) \
        + np.multiply(dvdx, dvdx) + np.multiply(dvdy, dvdy) + np.multiply(dvdz, dvdz) \
        + np.multiply(dwdx, dwdx) + np.multiply(dwdy, dwdy) + np.multiply(dwdz, dwdz)
    epsilon_tilde_base = visc*epsilon_tilde_base

    # 2) visc*(du_i/dx_j)*(du_j/dx_i)
    epsilon_2tilde_base = np.multiply(dudx, dudx) + np.multiply(dvdy, dvdy) + np.multiply(dwdz, dwdz) \
        + 2.*np.multiply(dudy, dvdx) + 2.*np.multiply(dudz, dwdx) + 2.*np.multiply(dvdz, dwdy)
    epsilon_2tilde_base = visc*epsilon_2tilde_base

    # 3) incompressible dissipation
    epsilon_incomp_base = epsilon_tilde_base + epsilon_2tilde_base

    # 4) compressible part of dissipation
    epsilon_comp_base = visc*term2

    # 4) full dissipation
    epsilon_base = epsilon_incomp_base + epsilon_comp_base

    # Integrate in vertical direction
    epsilon_int = trapezoid_integration(epsilon_base, y)
    epsilon_int_check = trapezoid_integration(epsilon_check, y)
    print("NEW (baseflow): ----> epsilon_int, epsilon_check = ", epsilon_int, epsilon_int_check)
    print("")

    #1/Re*( np.multiply(Uy, Uy) )

def compute_disturbance_dissipation(ueig, veig, weig, peig, reig, y, alpha, beta, D1, D2, bsfl, bsfl_ref, mob):

    testing = 0

    ny = len(ueig)

    if testing == 1:
        print("")
        print("---------------> testing in compute_disturbance_dissipation")
        print("")

    if (mob.boussinesq == -2 or mob.boussinesq == 10):
        visc = bsfl_ref.muref/bsfl_ref.rhoref
    else:
        visc = 1/bsfl_ref.Re
    
    # If non-dimensional equations, visc = 1/Re

    # Here I compute the dissipation (not the pseudo dissipation)
    
    # The vertical direction is z

    # Get some quantities to compute the balances
    Mu, Mup, Rho, Rhop, Rhopp, rho2, rho3, rho_inv, rho2_inv, rho3_inv = get_baseflow_and_derivatives(bsfl, mob)
    ia, ib, ia2, ib2, iab, Du, Dv, Dw, Dr, D2u, D2v, D2w, D2r = get_eigenfunctions_quantities(alpha, beta, D1, D2, ueig, veig, weig, reig)

    Re = bsfl_ref.Re
    Sc = bsfl_ref.Sc

    if (testing == 1):

        nz = len(ueig)

        dudx = np.random.rand(nz) + 1j*np.random.rand(nz)
        dudy = np.random.rand(nz) + 1j*np.random.rand(nz)
        dudz = np.random.rand(nz) + 1j*np.random.rand(nz)
        
        dvdx = np.random.rand(nz) + 1j*np.random.rand(nz)
        dvdy = np.random.rand(nz) + 1j*np.random.rand(nz)
        dvdz = np.random.rand(nz) + 1j*np.random.rand(nz)
        
        dwdx = np.random.rand(nz) + 1j*np.random.rand(nz)
        dwdy = np.random.rand(nz) + 1j*np.random.rand(nz)
        dwdz = np.random.rand(nz) + 1j*np.random.rand(nz)

    else:

        dudx,dudy,dudz,dvdx,dvdy,dvdz,dwdx,dwdy,dwdz,dpdx,dpdy,dpdz,drdx,drdy,drdz = get_eigenfunction_derivatives(ueig, veig, weig, peig, reig, alpha, beta, D1)
        
    sub1 = dudy+dvdx
    sub2 = dvdz+dwdy
    sub3 = dudz+dwdx

    div  = dudx+dvdy+dwdz
    
    term1 = 2.*( compute_inner_prod(dudx, dudx) + compute_inner_prod(dvdy, dvdy) + compute_inner_prod(dwdz, dwdz) )
    term2 = -2./3.*compute_inner_prod(div, div)
    term3 = compute_inner_prod(sub1, sub1) + compute_inner_prod(sub2, sub2) + compute_inner_prod(sub3, sub3)

    epsilon_check = visc*( term1 + term2 + term3 )

    ########################################
    # Various components of the dissipation
    ########################################

    # 1) visc*(du_i/dx_j)*(du_i/dx_j)
    epsilon_tilde_dist = compute_inner_prod(dudx, dudx) + compute_inner_prod(dudy, dudy) + compute_inner_prod(dudz, dudz) \
                       + compute_inner_prod(dvdx, dvdx) + compute_inner_prod(dvdy, dvdy) + compute_inner_prod(dvdz, dvdz) \
                       + compute_inner_prod(dwdx, dwdx) + compute_inner_prod(dwdy, dwdy) + compute_inner_prod(dwdz, dwdz)
    epsilon_tilde_dist = visc*epsilon_tilde_dist

    # 2) visc*(du_i/dx_j)*(du_j/dx_i)
    epsilon_2tilde_dist = compute_inner_prod(dudx, dudx) + compute_inner_prod(dvdy, dvdy) + compute_inner_prod(dwdz, dwdz) \
                        + 2.*compute_inner_prod(dudy, dvdx) + 2.*compute_inner_prod(dudz, dwdx) + 2.*compute_inner_prod(dvdz, dwdy)
    epsilon_2tilde_dist = visc*epsilon_2tilde_dist

    # 3) incompressible dissipation
    epsilon_incomp_dist = epsilon_tilde_dist + epsilon_2tilde_dist

    # 4) compressible part of dissipation
    epsilon_comp_dist = visc*term2

    # 5) full dissipation
    epsilon_dist = epsilon_incomp_dist + epsilon_comp_dist 

    # Compute dissipation for b-equation
    vp = -np.divide(reig, rho2)
    epsilon_b = np.divide(compute_inner_prod(vp, div), rho2)
    
    epsilon_b_tilde = 2.*np.multiply( rho2_inv, compute_inner_prod(ia*reig, ia*reig) + \
                                            compute_inner_prod(ib*reig, ib*reig) + \
                                            compute_inner_prod(Dr, Dr) )

    if( visc != 1/(Re*Sc) ):
        print("visc = ", visc)
        print("1/(Re*Sc) = ", 1/(Re*Sc))
        input("Check your viscosity in compute_disturbance_dissipation")
    
    epsilon_b_tilde = 1/(Re*Sc)*epsilon_b_tilde
    
    # Disturbance kinetic energy
    KE_dist = compute_disturbance_kin_energy(ueig, veig, weig)

    # Density self correlation b
    b = np.divide( compute_inner_prod(reig, reig), rho2)

    # Check that quantities are real
    Check_eps  = np.max(np.abs(epsilon_dist.imag))
    Check_epsb = np.max(np.abs(epsilon_b.imag))
    Check_KE   = np.max(np.abs(KE_dist.imag))
    Check_b    = np.max(np.abs(b.imag))

    tol = 1e-20
    
    if ( Check_eps < tol and Check_epsb < tol and Check_KE < tol and Check_b < tol ):
        KE_dist = KE_dist.real
        epsilon_b = epsilon_b.real
        epsilon_dist = epsilon_dist.real
        b = b.real
    else:
        print("np.max(np.abs(b.imag))            = ", np.max(np.abs(b.imag)))
        print("np.max(np.abs(epsilon_dist.imag)) = ", np.max(np.abs(epsilon_dist.imag)))
        print("np.max(np.abs(epsilon_b.imag))    = ", np.max(np.abs(epsilon_b.imag)))
        print("np.max(np.abs(KE_dist.imag))      = ", np.max(np.abs(KE_dist.imag)))
        print("")
        sys.exit("Error: some quantities have a nonzero imaginary part")
        #

    # Integrate in vertical direction
    epsilon_int = trapezoid_integration(epsilon_dist, y)
    epsilon_int_check = trapezoid_integration(epsilon_check, y)
    #print("NEW (disturbance): ----> epsilon_int, epsilon_check = ", epsilon_int, epsilon_int_check)
    
    Plot_dissip_compos = 0
    if (Plot_dissip_compos==1):
        plot_disturbance_dissipation(y, epsilon_dist, epsilon_incomp_dist, epsilon_comp_dist, epsilon_tilde_dist, epsilon_2tilde_dist, epsilon_b, epsilon_b_tilde)

    #print("epsilon_dist = ", epsilon_dist)
    
    return KE_dist, epsilon_dist, epsilon_tilde_dist-epsilon_2tilde_dist


    # Plot diveregence dimensional/non-dimensional
    # --------------------------------------------

    # ptn = plt.gcf().number + 1

    
    # f = plt.figure(ptn)
    # ax = plt.subplot(111)
    # ax.plot(np.abs(dwdz),        y, 'k', linewidth=1.5, label=r"div (non-dm)")
    # ax.plot(np.abs(dwdz_nondim), y, 'g--', linewidth=1.5, label=r"div_non_dim")

    # plt.xlabel("dwdz (ABS CHECK HERE)", fontsize=14)
    # plt.ylabel('z', fontsize=16)
    # plt.gcf().subplots_adjust(left=0.18)
    # plt.gcf().subplots_adjust(bottom=0.15)
    # plt.ylim([ymin, ymax])

    # ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15),   # box with lower left corner at (x, y)
    #           ncol=2, fancybox=True, shadow=True, fontsize=10)
    
    # f.show()

    # input("HHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHH")

    #print("max(dudx), max(dvdy), max(dwdz) (nondimensional) = ", np.max(dudx), np.max(dvdy), np.max(dwdz))
    #print("max(div) in compute_disturbance_dissipation (nondimensional):", np.max(div))



def compute_a_eq_ener_bal_boussi(ueig, veig, weig, peig, reig, D1, D2, y, alpha, beta, omega_i, bsfl, bsfl_ref, mob, rt_flag):

    #plt.close('all') 
    
    # Get some quantities to compute the balances
    Mu, Mup, Rho, Rhop, Rhopp, rho2, rho3, rho_inv, rho2_inv, rho3_inv = get_baseflow_and_derivatives(bsfl, mob)
    ia, ib, ia2, ib2, iab, Du, Dv, Dw, Dr, D2u, D2v, D2w, D2r = get_eigenfunctions_quantities(alpha, beta, D1, D2, ueig, veig, weig, reig)
    
    Sc = bsfl_ref.Sc
    Re = bsfl_ref.Re
    Fr = bsfl_ref.Fr

    Lref = bsfl_ref.Lref
    Uref = bsfl_ref.Uref
    muref = bsfl_ref.muref
    rhoref = bsfl_ref.rhoref

    div = ( 1j*alpha*ueig + 1j*beta*veig + Dw )

    v = -np.divide(reig, rho2)
    Dp  = np.matmul(D1, peig)

    dwdx    = (1j*alpha)*weig
    dwdy    = (1j*beta)*weig
    dwdz    = Dw

    drdx    = (1j*alpha)*reig
    drdy    = (1j*beta)*reig
    drdz    = Dr

    d2wdx2  = (1j*alpha)**2.*weig
    d2wdy2  = (1j*beta)**2.*weig
    d2wdz2  = D2w

    d2rdx2  = (1j*alpha)**2.*reig
    d2rdy2  = (1j*beta)**2.*reig
    d2rdz2  = D2r

    # LHS for a-equation
    LHS_aEq = 2*omega_i*np.divide(compute_inner_prod(reig, weig), Rho)
    
    # Compute Laplacian of a
    Lap_bsfl1 = -2.*np.multiply(rho2_inv, Rhop)
    Lap_t1 = np.multiply(Lap_bsfl1, compute_inner_prod(reig, Dw))
    Lap_t2 = np.multiply(Lap_bsfl1, compute_inner_prod(weig, Dr))

    Lap_t3 = np.multiply(rho_inv, compute_inner_prod(reig, d2wdx2 + d2wdy2 + d2wdz2))
    Lap_t4 = np.multiply(rho_inv, compute_inner_prod(weig, d2rdx2 + d2rdy2 + d2rdz2))

    Lap_t5 = 2.*np.multiply(rho_inv, compute_inner_prod(drdx, dwdx) + compute_inner_prod(drdy, dwdy), compute_inner_prod(Dr, Dw))

    Lap_bsfl6 = 2.*np.multiply(rho3_inv, np.multiply(Rhop, Rhop))
    Lap_t6 = np.multiply(Lap_bsfl6, compute_inner_prod(reig, weig))

    Lap_bsfl7 = -np.multiply(rho2_inv, Rhopp)
    Lap_t7 = np.multiply(Lap_bsfl7, compute_inner_prod(reig, weig))
    
    Lap_a = Lap_t1 + Lap_t2 + Lap_t3 + Lap_t4 + Lap_t5 + Lap_t6 + Lap_t7

    # Compute RHS
    RHS_a_t1 = np.multiply(Rho, compute_inner_prod(v, Dp) )
    RHS_a_t2 = -np.multiply(rho_inv, np.multiply(Rhop, compute_inner_prod(weig, weig)))

    RHS_a_bsfl3 = 2.*np.multiply(rho2_inv, Rhop)
    RHS_a_t3 = np.multiply(RHS_a_bsfl3, compute_inner_prod(reig, Dw) + compute_inner_prod(weig, Dr))

    RHS_a_bsfl33 = -2.*np.multiply(rho3_inv, np.multiply(Rhop, Rhop))
    RHS_a_t33 = np.multiply(RHS_a_bsfl33, compute_inner_prod(reig, weig))

    RHS_a_bsfl4 = np.multiply(rho2_inv, Rhopp)
    RHS_a_t4 = np.multiply(RHS_a_bsfl4, compute_inner_prod(reig, weig))

    RHS_a_bsfl5 = -2.*np.multiply(rho_inv, 1.)
    RHS_a_t5 = np.multiply(RHS_a_bsfl5, compute_inner_prod(drdx, dwdx) + compute_inner_prod(drdy, dwdy), compute_inner_prod(Dr, Dw) )

    
    RHS_a_bsfl6 = -np.multiply(rho_inv, 1.)
    RHS_a_t6 = np.multiply(RHS_a_bsfl6, compute_inner_prod(weig, d2rdx2+d2rdy2+d2rdz2))

    RHS_a_bsfl7 = -np.multiply(rho_inv, 1.)/Sc
    RHS_a_t7 = np.multiply(RHS_a_bsfl7, compute_inner_prod(reig, d2wdx2+d2wdy2+d2wdz2))
    
    RHS_a_t8 = -np.multiply(rho_inv, compute_inner_prod(reig, reig))/Fr**2.

    RHS_aEq = RHS_a_t1 + RHS_a_t2 + (Sc + 1.)/(Re*Sc)*( Lap_a + RHS_a_t3 + RHS_a_t33 + RHS_a_t4 + RHS_a_t5 ) + 1./Re*( RHS_a_t6 + RHS_a_t7 ) + RHS_a_t8 

    energy_bal_aEq  = np.amax(np.abs(LHS_aEq-RHS_aEq))
    print("Energy balance (RT) a-equation (BOUSSINESQ) = ", energy_bal_aEq)


def compute_a_equation_energy_balance(ueig, veig, weig, peig, reig, D1, D2, y, alpha, beta, omega_i, bsfl, bsfl_ref, mob, rt_flag):

    #plt.close('all') 
    
    # Get some quantities to compute the balances
    Mu, Mup, Rho, Rhop, Rhopp, rho2, rho3, rho_inv, rho2_inv, rho3_inv = get_baseflow_and_derivatives(bsfl, mob)
    ia, ib, ia2, ib2, iab, Du, Dv, Dw, Dr, D2u, D2v, D2w, D2r = get_eigenfunctions_quantities(alpha, beta, D1, D2, ueig, veig, weig, reig)
    
    Sc = bsfl_ref.Sc
    Re = bsfl_ref.Re
    Fr = bsfl_ref.Fr

    Lref = bsfl_ref.Lref
    Uref = bsfl_ref.Uref
    muref = bsfl_ref.muref
    rhoref = bsfl_ref.rhoref

    div = ( 1j*alpha*ueig + 1j*beta*veig + Dw )

    omega_i_dim = omega_i*Uref/Lref

    d2udxdz = 1j*alpha*Du
    d2wdx2  = (1j*alpha)**2.*weig
    d2vdydz = 1j*beta*Dv
    d2wdy2  = (1j*beta)**2.*weig
    d2wdz2  = D2w

    dPdz = -Rho*1./Fr**2.
    dpprimedz = np.matmul(D1, peig)

    # NONDIM BALANCE

    b_ = np.divide(compute_inner_prod(reig, reig), rho2)
    v_ = -np.divide(reig, rho2)

    LHS_aEq_non_dim = 2*omega_i*compute_inner_prod(reig, weig)

    ddivdz = ( d2udxdz + d2vdydz + d2wdz2 )

    #print("ddivdz_dim = ", ddivdz_dim)
    
    dTau13primedx = np.multiply(1/Re, d2udxdz + d2wdx2)
    dTau23primedy = np.multiply(1/Re, d2vdydz + d2wdy2)
    dTau33primedz = np.multiply(1/Re, 2.*D2w - 2./3.*ddivdz)

    pres_term_non_dim = compute_inner_prod(v_, dpprimedz)
    visc_term_non_dim = compute_inner_prod(v_, -dTau13primedx -dTau23primedy -dTau33primedz)
    
    R1_non_dim = np.multiply(b_, dPdz)
    R2_non_dim = -np.multiply(compute_inner_prod(weig, weig), Rhop)
    R3_non_dim = np.multiply(Rho, pres_term_non_dim)
    R4_non_dim = np.multiply(Rho, visc_term_non_dim)
    R5_non_dim = -np.multiply(Rho, compute_inner_prod(weig, div))
    RHS_aEq_non_dim = R1_non_dim + R2_non_dim + R3_non_dim + R4_non_dim + R5_non_dim
    
    energy_bal_aEq_non_dim  = np.amax(np.abs(LHS_aEq_non_dim-RHS_aEq_non_dim))
    print("Energy balance (RT) a-equation ---- NONDIM = ", energy_bal_aEq_non_dim)

    # Make eigenfunctions dimensional
    ueig = ueig*Uref
    veig = veig*Uref
    weig = weig*Uref

    peig = peig*rhoref*Uref**2.
    reig = reig*rhoref

    # Compute dimensional baseflow
    Rho_dim = Rho*rhoref
    Rhop_dim = Rhop*rhoref/Lref

    Rho2_dim = np.multiply(Rho_dim, Rho_dim)

    Mu_dim = Mu*muref

    #print("Mu_dim = ", Mu_dim)

    # Baseflow pressure gradient
    dPdz_dim = dPdz*rhoref*Uref**2./Lref

    #print("dPdz_dim = ", dPdz_dim)
    #print("")
    #print("Rho_dim*9.81 = ", Rho_dim*9.81)

    # Dimensional b
    b_dim = np.divide(compute_inner_prod(reig, reig), Rho2_dim)
    v_dim = -np.divide(reig, Rho2_dim)

    # LHS of a-Equation i=3 =====> third component of a-Equation
    LHS_aEq = 2*omega_i_dim*compute_inner_prod(reig, weig)

    # RHS of a-Equation
    scale = Uref/Lref**2.
    
    d2udxdz_dim = d2udxdz*scale
    d2wdx2_dim  = d2wdx2*scale
    d2vdydz_dim = d2vdydz*scale
    d2wdy2_dim  = d2wdy2*scale
    d2wdz2_dim  = d2wdz2*scale

    D2w_dim = D2w*scale

    div_dim = div*Uref/Lref
    ddivdz_dim = ( d2udxdz_dim + d2vdydz_dim + d2wdz2_dim )

    #print("ddivdz_dim = ", ddivdz_dim)
    
    dTau13primedx_dim = np.multiply(Mu_dim, d2udxdz_dim + d2wdx2_dim)
    dTau23primedy_dim = np.multiply(Mu_dim, d2vdydz_dim + d2wdy2_dim)
    dTau33primedz_dim = np.multiply(Mu_dim, 2.*D2w_dim - 2./3.*ddivdz_dim)

    dpprimedz_dim = dpprimedz*rhoref*Uref**2./Lref

    pres_term = compute_inner_prod(v_dim, dpprimedz_dim)
    visc_term = compute_inner_prod(v_dim, -dTau13primedx_dim -dTau23primedy_dim -dTau33primedz_dim)

    LHS_aEq = 2*omega_i_dim*compute_inner_prod(reig, weig)
    
    R1 = np.multiply(b_dim, dPdz_dim)
    R2 = -np.multiply(compute_inner_prod(weig, weig), Rhop_dim)
    R3 = np.multiply(Rho_dim, pres_term)
    R4 = np.multiply(Rho_dim, visc_term)
    R5 = -np.multiply(Rho_dim, compute_inner_prod(weig, div_dim))
    RHS_aEq = R1 + R2 + R3 + R4 + R5

    energy_bal_aEq  = np.amax(np.abs(LHS_aEq-RHS_aEq))
    print("Energy balance (RT) a-equation       = ", energy_bal_aEq)
    
    #print("LHS_aEq = ", LHS_aEq)

    #print("R1 = ", R1)
    #print("R2 = ", R2)
    #print("R3 = ", R3)
    #print("R4 = ", R4)
    #print("R5 = ", R5)

    ymin = -0.02
    ymax = -ymin

    ptn = plt.gcf().number + 1
    
    f = plt.figure(ptn)
    ax = plt.subplot(111)
    
    plt.plot(np.real(R1), y, 'b--', linewidth=1.5, label=r"$b\dfrac{\partial \overline{p}}{\partial z}$")
    plt.plot(np.real(R2), y, 'g', linewidth=1.5, label=r"$-\overline{w'w'}\dfrac{\partial \overline{\rho}}{\partial z}$")
    plt.plot(np.real(R3), y, 'b', linewidth=1.5, label=r"$\overline{\rho}\overline{v'\dfrac{\partial p'}{\partial z}}$")    
    plt.plot(np.real(R4), y, 'm', linewidth=1.5, label=r"$-\overline{\rho}\overline{v'\dfrac{\partial \tau'_{ni}}{\partial x_n}}$")    
    plt.plot(np.real(R5), y, 'c--', linewidth=1.5, label=r"$-\overline{\rho}\overline{w'\dfrac{\partial u'_i}{\partial x_i}}$")
    plt.plot(np.real(RHS_aEq), y, 'r', linewidth=1.5, label="RHS")

    plt.plot(np.real(LHS_aEq), y, 'k--', linewidth=1.5, label=r"LHS: $\dfrac{\partial \left( \overline{\rho} a_3 \right)}{\partial t}$")
    
    plt.xlabel("a-equation terms", fontsize=14)
    plt.ylabel('y', fontsize=16)
    plt.gcf().subplots_adjust(left=0.16)
    plt.gcf().subplots_adjust(bottom=0.15)
    plt.ylim([ymin, ymax])

    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=4, fancybox=True, shadow=True, fontsize=10)

    #plt.legend(loc="upper right")
    #plt.legend(fontsize=10)#, ncol = 7)
    
    f.show()


    ptn = ptn + 1
    
    f = plt.figure(ptn)
    ax = plt.subplot(111)
    ax.plot(np.real(R1_non_dim), y, 'b--', linewidth=1.5, label=r"$b\dfrac{\partial \overline{p}}{\partial z}$")
    ax.plot(np.real(R2_non_dim), y, 'g', linewidth=1.5, label=r"$-\overline{w'w'}\dfrac{\partial \overline{\rho}}{\partial z}$")
    ax.plot(np.real(R3_non_dim), y, 'b', linewidth=1.5, label=r"$\overline{\rho}\overline{v'\dfrac{\partial p'}{\partial z}}$")    
    ax.plot(np.real(R4_non_dim), y, 'm', linewidth=1.5, label=r"$-\overline{\rho}\overline{v'\dfrac{\partial \tau'_{ni}}{\partial x_n}}$")    
    ax.plot(np.real(R5_non_dim), y, 'c--', linewidth=1.5, label=r"$-\overline{\rho}\overline{w'\dfrac{\partial u'_i}{\partial x_i}}$")

    ax.plot(np.real(RHS_aEq_non_dim), y, 'r', linewidth=1.5, label="RHS")

    ax.plot(np.real(LHS_aEq_non_dim), y, 'k--', linewidth=1.5, label=r"LHS: $\dfrac{\partial \left( \overline{\rho} a_3 \right)}{\partial t}$")
    
    plt.xlabel("a-equation terms ---- NONDIMENSIONAL", fontsize=14)
    plt.ylabel('y', fontsize=16)
    plt.gcf().subplots_adjust(left=0.16)
    plt.gcf().subplots_adjust(bottom=0.15)
    plt.ylim([ymin, ymax])

    #plt.legend(bbox_to_anchor=(1.05, 1.05), ncol=4, borderaxespad=0)
    #plt.legend(loc="upper right")
    #plt.legend(fontsize=8)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15),   # box with lower left corner at (x, y)
              ncol=4, fancybox=True, shadow=True, fontsize=10)
    
    f.show()


    input("Final last check")


def compute_a_eq_ener_bal_sandoval(ueig, veig, weig, peig, reig, D1, D2, y, alpha, beta, omega_i, bsfl, bsfl_ref, mob, rt_flag):

    # Get some quantities to compute the balances
    Mu, Mup, Rho, Rhop, Rhopp, rho2, rho3, rho_inv, rho2_inv, rho3_inv = get_baseflow_and_derivatives(bsfl, mob)
    ia, ib, ia2, ib2, iab, Du, Dv, Dw, Dr, D2u, D2v, D2w, D2r = get_eigenfunctions_quantities(alpha, beta, D1, D2, ueig, veig, weig, reig)
    
    Sc = bsfl_ref.Sc
    Re = bsfl_ref.Re
    Fr = bsfl_ref.Fr

    Lref = bsfl_ref.Lref
    Uref = bsfl_ref.Uref
    muref = bsfl_ref.muref
    rhoref = bsfl_ref.rhoref

    div = ( 1j*alpha*ueig + 1j*beta*veig + Dw )
    Dp  = np.matmul(D1, peig)

    d2udxdz = 1j*alpha*Du
    d2vdydz = 1j*beta*Dv
    
    d2wdx2  = (1j*alpha)**2.*weig
    d2wdy2  = (1j*beta)**2.*weig
    d2wdz2  = D2w

    # Check that 2w'*z-momentum equation is satisfied
    LHS_wwEq = 2*omega_i*np.multiply(compute_inner_prod(weig, weig), Rho)

    rhs_wwt1 = -2.*compute_inner_prod(weig, Dp)
    
    rhs_wwt2 = 2./Re*( compute_inner_prod(weig, d2wdx2 + d2wdy2 + d2wdz2 ) + 1./3.*compute_inner_prod(weig, d2udxdz + d2vdydz + d2wdz2 ) )
    rhs_wwt3 = -2.*compute_inner_prod(reig, weig)/Fr**2.
    
    RHS_wwEq = rhs_wwt1 + rhs_wwt2 + rhs_wwt3
    energy_bal_wwEq  = np.amax(np.abs(LHS_wwEq-RHS_wwEq))
    print("Energy balance (RT) ww-equation (SANDOVAL) = ", energy_bal_wwEq)
    print("")

    # Check mass real quick
    LHS_massEq  = 2.*omega_i*compute_inner_prod(reig, reig)

    # Compute baseflow divergence
    dundxn_baseflow = 1/(Re*Sc)*( np.multiply(rho2_inv, np.multiply(Rhop, Rhop) ) - np.multiply(rho_inv, Rhopp) )
    
    rhs_mass_t1 = -2.*np.multiply(Rho, compute_inner_prod(reig, div))
    rhs_mass_t2 = -2.*np.multiply(Rhop, compute_inner_prod(reig, weig))
    rhs_mass_t3 = -2.*np.multiply(dundxn_baseflow, compute_inner_prod(reig, reig))
    
    RHS_massEq  = rhs_mass_t1 + rhs_mass_t2 + rhs_mass_t3

    mass_bal    = np.amax(np.abs(LHS_massEq-RHS_massEq))
    print("mass balance (RT) real quick (SANDOVAL) = ", mass_bal)
    print("")

    # LHS of a-Equation i=3 =====> third component of a-Equation
    LHS_aEq = 2*omega_i*np.divide(compute_inner_prod(reig, weig), Rho)
    #LHS_aEq = np.multiply(Rho, LHS_aEq)

    # RHS of a-Equation
    rhs_t1 = -np.multiply(rho2_inv, compute_inner_prod(reig, Dp))
    rhs_t2 = np.multiply(rho2_inv/Re, compute_inner_prod(reig, d2wdx2 + d2wdy2 + d2wdz2 ) + 1./3.*compute_inner_prod(reig, d2udxdz + d2vdydz + d2wdz2 ) )

    rhs_t3 = -np.multiply(rho2_inv, compute_inner_prod(reig, reig))/Fr**2.
    rhs_t4 = -np.multiply( 1., compute_inner_prod(div, weig) )
    rhs_t5 = -np.multiply(np.multiply(rho_inv, Rhop), compute_inner_prod(weig, weig))

    # term with baseflow divergence
    rhs_t6 = -np.multiply( np.multiply(rho_inv, dundxn_baseflow), compute_inner_prod(reig, weig) )
    
    RHS_aEq = rhs_t1 + rhs_t2 + rhs_t3 + rhs_t4 + rhs_t5 + rhs_t6
    energy_bal_aEq  = np.amax(np.abs(LHS_aEq-RHS_aEq))
    print("Energy balance (RT) a-equation (SANDOVAL) = ", energy_bal_aEq)
    print("")


    ptn = plt.gcf().number + 1
    
    f = plt.figure(ptn)
    ax = plt.subplot(111)
    
    plt.plot(np.real(RHS_aEq), y, 'r', linewidth=1.5, label="RHS (a-eq.)")
    plt.plot(np.real(LHS_aEq), y, 'k--', linewidth=1.5, label=r"LHS (a-eq.)")
    
    plt.xlabel('a-equation =========> SANDOVAL', fontsize=14)
    plt.ylabel('y', fontsize=16)
    plt.gcf().subplots_adjust(left=0.16)
    plt.gcf().subplots_adjust(bottom=0.15)

    ymin = -0.02; ymax = -ymin
    plt.ylim([ymin, ymax])

    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=4, fancybox=True, shadow=True, fontsize=10)

    f.show()


def compute_b_equation_energy_balance(ueig, veig, weig, peig, reig, D1, D2, y, alpha, beta, omega_i, bsfl, bsfl_ref, mob, rt_flag):

    #plt.close('all')

    #print("omega_i = ", omega_i)
    
    # Get some quantities to compute the balances
    Mu, Mup, Rho, Rhop, Rhopp, rho2, rho3, rho_inv, rho2_inv, rho3_inv = get_baseflow_and_derivatives(bsfl, mob)
    ia, ib, ia2, ib2, iab, Du, Dv, Dw, Dr, D2u, D2v, D2w, D2r = get_eigenfunctions_quantities(alpha, beta, D1, D2, ueig, veig, weig, reig)
    
    Sc = bsfl_ref.Sc
    Re = bsfl_ref.Re
    Fr = bsfl_ref.Fr

    Lref = bsfl_ref.Lref
    Uref = bsfl_ref.Uref
    muref = bsfl_ref.muref
    rhoref = bsfl_ref.rhoref

    #
    # Compute non-dimensional b-equation balance
    #

    LHS_bEq = 2*omega_i*np.multiply(rho2_inv, compute_inner_prod(reig, reig))

    # Baseflow w-velocity for R-T
    wvel_base = -1/(Re*Sc)*np.multiply(rho_inv, Rhop)

    drhodz2_base = np.multiply(Rhop, Rhop)

    term1_base = 1/(Re*Sc)*( np.multiply(rho2_inv, drhodz2_base) )
    term2_base = -1/(Re*Sc)*np.multiply(rho_inv, Rhopp)
    
    div_base = term1_base + term2_base

    # Note this one is assuming a steady LST baseflow
    #w_vel_base_lst_steady = -np.divide( np.multiply(Rho, div_base), Rhop)
    w_vel_base_lst_steady_exact = 1/(Re*Sc)*( -np.multiply(rho_inv, Rhop) -2.*y/bsfl.delta**2.)

    opt_vel = 1

    if (opt_vel==1):
        wvel_base = w_vel_base_lst_steady_exact
    else:
        wvel_base = wvel_base

    v = -np.divide(reig, rho2)
    div = ( 1j*alpha*ueig + 1j*beta*veig + Dw )
    
    conv1_bsfl = -2.*np.multiply(wvel_base, rho2_inv)
    conv2_bsfl = 2.*np.multiply( wvel_base, np.multiply(rho3_inv, Rhop) )

    RHS_conv = np.multiply(conv1_bsfl, compute_inner_prod(reig, Dr)) + np.multiply(conv2_bsfl, compute_inner_prod(reig, reig))

    prod_bsfl = -2.*np.multiply(rho2_inv, Rhop)
    RHS_prod = np.multiply(prod_bsfl, compute_inner_prod(reig, weig))
    RHS_div = -2.*np.multiply(rho_inv, compute_inner_prod(reig, div))
    
    RHS_bEq = RHS_prod + RHS_div + RHS_conv #+ 2.*np.multiply(Rho, compute_inner_prod(v, div))

    ymin = -0.002
    ymax = -ymin

    ptn = plt.gcf().number + 1
    
    f = plt.figure(ptn)
    ax = plt.subplot(111)
    
    plt.plot(np.real(RHS_conv), y, 'b', linewidth=1.5, label=r"RHS_conv")
    plt.plot(np.real(RHS_prod), y, 'gs', markerfacecolor='none', markevery=3, linewidth=1.5, label=r"RHS_prod")
    plt.plot(np.real(RHS_div), y, 'r--', linewidth=1.5, label="RHS_div")
    plt.plot(np.real(RHS_bEq), y, 'k', linewidth=1.5, label="RHS")

    plt.plot(np.real(LHS_bEq), y, 'm--', linewidth=1.5, label=r"LHS")
    
    plt.xlabel("b-equation terms (NON-DIMENSIONAL)", fontsize=14)
    plt.ylabel('y', fontsize=16)
    plt.gcf().subplots_adjust(left=0.16)
    plt.gcf().subplots_adjust(bottom=0.15)
    plt.ylim([ymin, ymax])

    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=4, fancybox=True, shadow=True, fontsize=10)

    #plt.legend(loc="upper right")
    #plt.legend(fontsize=10)#, ncol = 7)
    
    f.show()

    # SOME CHECKS
    t1_base = 2.*np.multiply( Rhop, np.multiply(Rhop, rho3_inv))
    t2_base = -2.*np.multiply( rho2_inv, Rhop)
    t3_base = np.multiply( rho_inv, 1.0)
    t4_base = -np.multiply( rho2_inv, Rhopp)

    should_be_equal_to_rho_dist_times_div_dist = -1/(Re*Sc)*( np.multiply(t1_base, compute_inner_prod(reig, reig) )+ \
                                                              np.multiply(t2_base, compute_inner_prod(reig, Dr)   )+ \
                                                              np.multiply(t3_base, compute_inner_prod(reig, ia2*reig)  )+ \
                                                              np.multiply(t3_base, compute_inner_prod(reig, ib2*reig)  )+ \
                                                              np.multiply(t3_base, compute_inner_prod(reig, D2r)  )+ \
                                                              np.multiply(t4_base, compute_inner_prod(reig, reig) )   )

    rhod_dis_times_div_dist = compute_inner_prod(reig, div)

    #print("should_be_equal_to_rho_dist_times_div_dist = ", should_be_equal_to_rho_dist_times_div_dist)
    #print("")
    #print("rhod_dis_times_div_dist = ", rhod_dis_times_div_dist)
                                    
    ptn = plt.gcf().number + 1
    
    f = plt.figure(ptn)
    ax = plt.subplot(111)
    
    plt.plot(np.real(should_be_equal_to_rho_dist_times_div_dist), y, 'b', linewidth=1.5, label=r"should_be_equal_to_rho_dist_times_div_dist")
    plt.plot(np.real(rhod_dis_times_div_dist), y, 'r--', linewidth=1.5, label=r"rhod_dis_times_div_dist")
    
    plt.xlabel("SOME CHECKS (NON-DIMENSIONAL)", fontsize=14)
    plt.ylabel('y', fontsize=16)
    plt.gcf().subplots_adjust(left=0.16)
    plt.gcf().subplots_adjust(bottom=0.15)
    plt.ylim([ymin, ymax])

    #ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=4, fancybox=True, shadow=True, fontsize=10)

    f.show()
    
    input("check b-eq energy balance now!!!!")



    
    
    energy_bal_bEq  = np.amax(np.abs(LHS_bEq-RHS_bEq))
    print("Energy balance (RT) b-equation NONDIM       = ", energy_bal_bEq)
    
    # Dimensional growth rate
    omega_i_dim = omega_i*Uref/Lref

    # Make eigenfunctions dimensional
    ueig = ueig*Uref
    veig = veig*Uref
    weig = weig*Uref

    peig = peig*rhoref*Uref**2.
    reig = reig*rhoref

    # Compute dimensional baseflow
    Rho_dim = Rho*rhoref
    Rhop_dim = Rhop*rhoref/Lref

    Rho2_dim = np.multiply(Rho_dim, Rho_dim)

    Rho_dim_inv = 1/Rho_dim
    Rho2_dim_inv = 1/Rho2_dim

    Mu_dim = Mu*muref

    # Dimensional v (v=1/rho)
    v_dim = -np.divide(reig, Rho2_dim)

    div_dim = div*Uref/Lref

    #print("div = ", div)
    #print("")
    #print("div_dim = ", div_dim)
    
    # b-Equation balance
    LHS_bEq = 2*omega_i_dim*np.multiply(Rho2_dim_inv, compute_inner_prod(reig, reig))

    bsfl_t1 = 2.*np.multiply(Rho2_dim_inv, Rhop_dim)
    R1 = -np.multiply(bsfl_t1, compute_inner_prod(reig, weig))
    R2 = 2.*np.multiply(Rho_dim, compute_inner_prod(v_dim, div_dim))
    RHS_bEq = R1 + R2

    #print("LHS_bEq = ", LHS_bEq)
    #print("")
    #print("")
    #print("RHS_bEq = ", RHS_bEq)
    energy_bal_bEq  = np.amax(np.abs(LHS_bEq-RHS_bEq))
    energy_bal_bEq222  = np.amax(np.abs(LHS_bEq[1:-2]-RHS_bEq[1:-2]))
    print("Energy balance (RT) b-equation       = ", energy_bal_bEq)
    print("Energy balance (RT) b-equation 222   = ", energy_bal_bEq222)

    # pv = p'*v' with v' being v'=-rho'/rho_bar^2
    dpv_distdz_dim = compute_inner_prod(peig, np.matmul(bsfl.D1_dim, v_dim)) + compute_inner_prod(v_dim, np.matmul(bsfl.D1_dim, peig))

    vdpdz_dim = compute_inner_prod(v_dim, np.matmul(bsfl.D1_dim, peig))
    pdvdz_dim = compute_inner_prod(peig, np.matmul(bsfl.D1_dim, v_dim))

    vdpdz_dim_check = dpv_distdz_dim - pdvdz_dim


    
    ptn = plt.gcf().number + 1
    
    f = plt.figure(ptn)
    ax = plt.subplot(111)
    
    plt.plot(np.real(dpv_distdz_dim), y, 'k', linewidth=1.5, label=r"$\dfrac{\partial \left( \overline{p' v'}\right)}{\partial z}$")
    plt.plot(np.real(vdpdz_dim), y, 'r', linewidth=1.5, label="$\overline{v'\dfrac{\partial p'}{\partial z}}$")
    plt.plot(np.real(pdvdz_dim), y, 'b', linewidth=1.5, label="$\overline{p'\dfrac{\partial v'}{\partial z}}$")
    plt.plot(np.real(vdpdz_dim_check), y, 'g--', linewidth=1.5, label="$\overline{v'\dfrac{\partial p'}{\partial z}}$ (CHECK)")
    
    plt.xlabel("pressure-specific-volume terms breakout", fontsize=14)
    plt.ylabel('y', fontsize=16)
    plt.gcf().subplots_adjust(left=0.16)
    plt.gcf().subplots_adjust(bottom=0.15)
    plt.ylim([ymin, ymax])

    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=4, fancybox=True, shadow=True, fontsize=10)

    #plt.legend(loc="upper right")
    #plt.legend(fontsize=10)#, ncol = 7)
    
    f.show()


    

    ptn = plt.gcf().number + 1
    
    f = plt.figure(ptn)
    ax = plt.subplot(111)
    
    plt.plot(np.real(R1), y, 'bs', markerfacecolor='none', markevery=2, linewidth=1.5, label=r"$-\dfrac{2}{\overline{\rho}^2} \dfrac{\partial \overline{\rho}}{\partial z} \overline{\rho'w'}$")
    plt.plot(np.real(R2), y, 'g', linewidth=1.5, label=r"$2\overline{\rho} \overline{ v'\dfrac{\partial u'_i}{\partial x_i}}$") ### R2 = 2.*np.multiply(Rho_dim, compute_inner_prod(v_dim, div_dim))
    plt.plot(np.real(RHS_bEq), y, 'r', linewidth=1.5, label="RHS")

    plt.plot(np.real(LHS_bEq), y, 'k--', linewidth=1.5, label=r"LHS: $\dfrac{\partial b}{\partial t}$")
    
    plt.xlabel("b-equation terms", fontsize=14)
    plt.ylabel('y', fontsize=16)
    plt.gcf().subplots_adjust(left=0.16)
    plt.gcf().subplots_adjust(bottom=0.15)
    plt.ylim([ymin, ymax])

    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=4, fancybox=True, shadow=True, fontsize=10)

    #plt.legend(loc="upper right")
    #plt.legend(fontsize=10)#, ncol = 7)
    
    f.show()

    input("After b-equation checks")



def compute_dsc_eq_bal_sandoval(ueig, veig, weig, peig, reig, D1, D2, y, alpha, beta, omega_i, bsfl, bsfl_ref, mob, rt_flag, map):

    # Get some quantities to compute the balances
    Mu, Mup, Rho, Rhop, Rhopp, rho2, rho3, rho_inv, rho2_inv, rho3_inv = get_baseflow_and_derivatives(bsfl, mob)
    ia, ib, ia2, ib2, iab, Du, Dv, Dw, Dr, D2u, D2v, D2w, D2r = get_eigenfunctions_quantities(alpha, beta, D1, D2, ueig, veig, weig, reig)
    
    Sc = bsfl_ref.Sc
    Re = bsfl_ref.Re
    Fr = bsfl_ref.Fr

    Lref = bsfl_ref.Lref
    Uref = bsfl_ref.Uref
    muref = bsfl_ref.muref
    rhoref = bsfl_ref.rhoref

    #
    # Compute non-dimensional density-self-correlation balance
    #

    LHS_dscEq = 2*omega_i*compute_inner_prod(reig, reig)

    prod_bsfl = -2.*Rhop
    RHS_prod = np.multiply(prod_bsfl, compute_inner_prod(reig, weig))
    
    RHS_visc1 = 1/(Re*Sc)*( 2.*compute_inner_prod(reig, ia2*reig) + 2.*compute_inner_prod(reig, ib2*reig) + 2.*compute_inner_prod(reig, D2r) )

    visc2_bsfl = 2.*np.multiply(rho2_inv, np.multiply(Rhop, Rhop))
    visc3_bsfl = -2.*np.multiply(rho_inv, Rhop)

    RHS_visc2 = 1/(Re*Sc)*( np.multiply(visc2_bsfl, compute_inner_prod(reig, reig)) )
    RHS_visc3 = 1/(Re*Sc)*( np.multiply(visc3_bsfl, 2*compute_inner_prod(reig, Dr)) )
    
    RHS_dscEq = RHS_prod + RHS_visc1 + RHS_visc2 + RHS_visc3

    energy_bal_dscEq  = np.amax(np.abs(LHS_dscEq-RHS_dscEq))
    print("Energy balance (RT) dsc-equation SANDOVAL (NON-DIMENSIONAL) = ", energy_bal_dscEq)


    
    ptn = plt.gcf().number + 1

    ymin = -0.002
    ymax = -ymin

    
    f = plt.figure(ptn)
    ax = plt.subplot(111)
    
    plt.plot(np.real(RHS_prod), y, 'k--', linewidth=1.5, label=r"RHS_prod")
    plt.plot(np.real(RHS_visc1), y, 'b--', linewidth=1.5, label=r"RHS_visc1")
    plt.plot(np.real(RHS_visc2), y, 'r--', linewidth=1.5, label=r"RHS_visc2")
    plt.plot(np.real(RHS_visc3), y, 'm--', linewidth=1.5, label=r"RHS_visc3")
    plt.plot(np.real(RHS_dscEq), y, 'k', linewidth=1.5, label=r"RHS")
    plt.plot(np.real(LHS_dscEq), y, 'g--', linewidth=1.5, label=r"LHS")
             
    plt.xlabel("dsc equation balance SANDOVAL (NON-DIMENSIONAL)", fontsize=14)
    plt.ylabel('y', fontsize=16)
    plt.gcf().subplots_adjust(left=0.16)
    plt.gcf().subplots_adjust(bottom=0.15)
    plt.ylim([ymin, ymax])

    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=3, fancybox=True, shadow=True, fontsize=10)

    f.show()





def compute_b_eq_bal_sandoval(ueig, veig, weig, peig, reig, D1, D2, y, alpha, beta, omega_i, bsfl, bsfl_ref, mob, rt_flag, map, KE_dist, epsilon_dist):

    # Get some quantities to compute the balances
    Mu, Mup, Rho, Rhop, Rhopp, rho2, rho3, rho_inv, rho2_inv, rho3_inv = get_baseflow_and_derivatives(bsfl, mob)
    ia, ib, ia2, ib2, iab, Du, Dv, Dw, Dr, D2u, D2v, D2w, D2r = get_eigenfunctions_quantities(alpha, beta, D1, D2, ueig, veig, weig, reig)

    rho4_inv = np.multiply(rho2_inv, rho2_inv)
    
    Sc = bsfl_ref.Sc
    Re = bsfl_ref.Re
    Fr = bsfl_ref.Fr

    Lref = bsfl_ref.Lref
    Uref = bsfl_ref.Uref
    muref = bsfl_ref.muref
    rhoref = bsfl_ref.rhoref

    v = -np.divide(reig, rho2)
    div = ( 1j*alpha*ueig + 1j*beta*veig + Dw )

    epsilon_b = compute_inner_prod(v, div)
    #
    # Compute non-dimensional density-self-correlation balance
    #

    form = 2

    LHS_bEq = 2*omega_i*np.multiply(rho2_inv, compute_inner_prod(reig, reig))

    prod_bsfl = -2.*np.multiply(rho2_inv, Rhop)
    RHS_prod = np.multiply(prod_bsfl, compute_inner_prod(reig, weig))

    if (form==1):
                
        visc1_bsfl = np.multiply(2., rho2_inv)
        visc2_bsfl = -4.*np.multiply(rho3_inv, Rhop)
        visc3_bsfl = 2.*np.multiply(rho4_inv, np.multiply(Rhop, Rhop))
        
        RHS_visc1 = 1/(Re*Sc)*( np.multiply(visc1_bsfl, compute_inner_prod(reig, ia2*reig) + compute_inner_prod(reig, ib2*reig) + compute_inner_prod(reig, D2r) ) )
        RHS_visc2 = 1/(Re*Sc)*( np.multiply(visc2_bsfl, compute_inner_prod(reig, Dr)) )
        RHS_visc3 = 1/(Re*Sc)*( np.multiply(visc3_bsfl, compute_inner_prod(reig, reig)) )

        RHS_bEq = RHS_prod + RHS_visc1 + RHS_visc2 + RHS_visc3

    elif(form==2):

        # Compute Laplacian of b (see leaflet on Laplacian of b)
        Lap_t1 = 1/(Re*Sc)*( 2.*np.multiply( rho2_inv, compute_inner_prod(reig, ia2*reig) + compute_inner_prod(reig, ib2*reig) + compute_inner_prod(reig, D2r) ) )

        epsilon_b_tilde = 1/(Re*Sc)*( 2.*np.multiply( rho2_inv, compute_inner_prod(ia*reig, ia*reig) + compute_inner_prod(ib*reig, ib*reig) + compute_inner_prod(Dr, Dr) ) )
        
        lap1_bsfl = -8.*np.multiply(rho3_inv, Rhop)
        lap2_bsfl = 6.*np.multiply(rho4_inv, np.multiply(Rhop, Rhop) )
        lap3_bsfl = -2.*np.multiply(rho3_inv, Rhopp)
        
        Lap_t3 = np.multiply(lap1_bsfl, compute_inner_prod(reig, Dr)) + np.multiply(lap2_bsfl, compute_inner_prod(reig, reig)) + np.multiply(lap3_bsfl, compute_inner_prod(reig, reig))
        Lap_t3 = 1/(Re*Sc)*Lap_t3

        Laplacian_b = Lap_t1 + epsilon_b_tilde + Lap_t3
        RHS_visc1 = Laplacian_b

        dbdz = 2.*np.multiply(rho2_inv, compute_inner_prod(reig, Dr)) - 2.*np.multiply(rho3_inv, np.multiply(Rhop, compute_inner_prod(reig, reig)))
        RHS_visc2 = 2/(Re*Sc)*np.multiply(rho_inv, np.multiply(Rhop, dbdz))

        visc3_bsfl = 2.*np.multiply(rho3_inv, Rhopp)
        RHS_visc3 = 1/(Re*Sc)*np.multiply(visc3_bsfl, compute_inner_prod(reig, reig))

        RHS_bEq = RHS_prod + Laplacian_b + RHS_visc2 + RHS_visc3 - epsilon_b_tilde

    else:
        sys.exit("Not a proper value for flag form!!!!!")
        
    energy_bal_bEq  = np.amax(np.abs(LHS_bEq-RHS_bEq))
    print("Energy balance (RT) b-equation SANDOVAL (NON-DIMENSIONAL) = ", energy_bal_bEq)

    
    ptn = plt.gcf().number + 1

    ymin = -0.002
    ymax = -ymin

    if (form==1):
    
        f = plt.figure(ptn)
        ax = plt.subplot(111)
        
        plt.plot(np.real(RHS_prod), y, 'k--', linewidth=1.5, label=r"RHS_prod")
        plt.plot(np.real(RHS_visc1), y, 'b--', linewidth=1.5, label=r"RHS_visc1")
        plt.plot(np.real(RHS_visc2), y, 'r--', linewidth=1.5, label=r"RHS_visc2")
        plt.plot(np.real(RHS_visc3), y, 'm--', linewidth=1.5, label=r"RHS_visc3")
        plt.plot(np.real(RHS_bEq), y, 'k', linewidth=1.5, label=r"RHS")
        plt.plot(np.real(LHS_bEq), y, 'g--', linewidth=1.5, label=r"LHS")
        
        plt.xlabel("b equation balance SANDOVAL (NON-DIMENSIONAL)", fontsize=14)
        plt.ylabel('y', fontsize=16)
        plt.gcf().subplots_adjust(left=0.16)
        plt.gcf().subplots_adjust(bottom=0.15)
        plt.ylim([ymin, ymax])
        
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=3, fancybox=True, shadow=True, fontsize=10)
        
        f.show()

    else:

        f = plt.figure(ptn)
        ax = plt.subplot(111)
        
        plt.plot(np.real(RHS_prod), y, 'k--', linewidth=1.5, label=r"Production")
        plt.plot(np.real(RHS_visc1), y, 'b--', linewidth=1.5, label=r"$\dfrac{1}{Re Sc}\nabla^2 b$")
        plt.plot(np.real(RHS_visc2), y, 'r--', linewidth=1.5, label=r"$\dfrac{1}{Re Sc} \dfrac{2}{\bar{\rho}}\dfrac{\partial \bar{\rho}}{\partial z}\dfrac{\partial b}{\partial z}$")
        plt.plot(np.real(RHS_visc3), y, 'm--', linewidth=1.5, label=r"$\dfrac{2b}{\bar{\rho}}\dfrac{\partial^2 \bar{\rho}}{\partial z^2}$")
        plt.plot(np.real(epsilon_b_tilde), y, 'c--', linewidth=1.5, label=r"$\tilde{\epsilon}_b$")
        plt.plot(np.real(RHS_bEq), y, 'k', linewidth=1.5, label=r"RHS")
        plt.plot(np.real(LHS_bEq), y, 'g--', linewidth=1.5, label=r"LHS")
        
        plt.xlabel("b equation balance SANDOVAL (NON-DIMENSIONAL)", fontsize=14)
        plt.ylabel('y', fontsize=16)
        plt.gcf().subplots_adjust(left=0.16)
        plt.gcf().subplots_adjust(bottom=0.15)
        plt.ylim([ymin, ymax])
        
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=4, fancybox=True, shadow=True, fontsize=10)
        
        f.show()

    # Plot exact and model dissipation

    epsilon_b_tilde = 1/(Re*Sc)*( 2.*np.multiply( rho2_inv, compute_inner_prod(ia*reig, ia*reig) + compute_inner_prod(ib*reig, ib*reig) + compute_inner_prod(Dr, Dr) ) )

    Cb1 = 2.
    b = np.multiply(rho2_inv, compute_inner_prod(reig, reig))

    Sturb = np.divide(KE_dist**1.5, epsilon_dist)
    t1_model = np.divide(np.multiply(KE_dist**0.5, b), np.multiply(Sturb, Rho))
    
    epsilon_b_model = Cb1/2.*t1_model

    cb1_t1 = np.divide(Sturb, np.multiply(KE_dist**0.5, b+1e-10))
    Cb1_lst = 2.*np.multiply(Rho, np.multiply(cb1_t1, compute_inner_prod(v, div)))

    
    ptn = plt.gcf().number + 1

    ymin = -0.002
    ymax = -ymin

    f = plt.figure(ptn)
    ax = plt.subplot(111)
    
    plt.plot(np.real(epsilon_b_tilde), y, 'k', linewidth=1.5, label=r"$\tilde{epsilon}_b$")
    plt.plot(np.real(epsilon_b_model), y, 'b--', linewidth=1.5, label=r"$\dfrac{C_{b1}}{2} \dfrac{\sqrt{K} b}{S \bar{\rho}}$")
    
    plt.xlabel("b equation balance SANDOVAL ----- Dissipation", fontsize=14)
    plt.ylabel('y', fontsize=16)
    plt.gcf().subplots_adjust(left=0.16)
    plt.gcf().subplots_adjust(bottom=0.15)
    plt.ylim([ymin, ymax])
    
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=4, fancybox=True, shadow=True, fontsize=10)
    
    f.show()


    # Plot Cb1 LST
    ptn = plt.gcf().number + 1

    ymin = -0.002
    ymax = -ymin

    f = plt.figure(ptn)
    ax = plt.subplot(111)
    
    plt.plot(np.real(Cb1_lst), y, 'k', linewidth=1.5, label=r"$C_{b1}=2 \overline{v'\dfrac{\partial u'_k}{\partial x_k}} \dfrac{S}{b \sqrt{K}}$")
    
    plt.xlabel("b equation balance SANDOVAL ---- $C_{b1} coefficient$", fontsize=14)
    plt.ylabel('y', fontsize=16)
    plt.gcf().subplots_adjust(left=0.16)
    plt.gcf().subplots_adjust(bottom=0.15)
    plt.ylim([ymin, ymax])
    
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=4, fancybox=True, shadow=True, fontsize=10)
    
    f.show()

    input("compute_b_eq_bal_sandoval")

def compute_dsc_eq_bal_boussi(ueig, veig, weig, peig, reig, D1, D2, y, alpha, beta, omega_i, bsfl, bsfl_ref, mob, rt_flag, map):

    # dsc: density-self-correlation

    ny = len(ueig)
    
    # Get some quantities to compute the balances
    Mu, Mup, Rho, Rhop, Rhopp, rho2, rho3, rho_inv, rho2_inv, rho3_inv = get_baseflow_and_derivatives(bsfl, mob)
    ia, ib, ia2, ib2, iab, Du, Dv, Dw, Dr, D2u, D2v, D2w, D2r = get_eigenfunctions_quantities(alpha, beta, D1, D2, ueig, veig, weig, reig)
    
    Sc = bsfl_ref.Sc
    Re = bsfl_ref.Re
    Fr = bsfl_ref.Fr

    Lref = bsfl_ref.Lref
    Uref = bsfl_ref.Uref
    Dref = bsfl_ref.Dref
    muref = bsfl_ref.muref
    rhoref = bsfl_ref.rhoref

    #
    # Compute non-dimensional density-self-correlation balance
    #

    # dsc --> density-self-correlation
    
    div = ( 1j*alpha*ueig + 1j*beta*veig + Dw )

    #print("")
    #print("In compute_density_self_correlation_equation_energy_balance_boussinesq, max. velocity divergence:")
    #print("max(div) (non-dimensional) = ", np.amax(np.abs(div)))
    
    LHS_dscEq = 2*omega_i*compute_inner_prod(reig, reig)

    # Note: not taking the triple correlation
    term_rhs1 = -2.*np.multiply(Rhop, compute_inner_prod(reig, weig))
    
    term_rhs2 = 2.*( compute_inner_prod(ia*reig, ia*reig) + compute_inner_prod(ib*reig, ib*reig) + compute_inner_prod(Dr, Dr) )
    term_rhs2 = term_rhs2 + 2.*( compute_inner_prod(reig, ia2*reig) + compute_inner_prod(reig, ib2*reig) + compute_inner_prod(reig, D2r) )
    term_rhs2 = 1/(Re*Sc)*term_rhs2

    # dissipation of rho'rho'
    term_rhs3 = -2.*( compute_inner_prod(ia*reig, ia*reig) + compute_inner_prod(ib*reig, ib*reig) + compute_inner_prod(Dr, Dr) )
    term_rhs3 =	1/(Re*Sc)*term_rhs3

    RHS_dscEq = term_rhs1 + term_rhs2 + term_rhs3
    
    energy_bal_dscEq  = np.amax(np.abs(LHS_dscEq-RHS_dscEq))
    print("Energy balance (RT) dsc-equation BOUSSINESQ (NON-DIMENSIONAL) = ", energy_bal_dscEq)

    #ymin = -0.005
    #ymax = -ymin

    # GET ymin-ymax
    ymin, ymax = FindResonableYminYmax(term_rhs3, y)
    
    ptn = plt.gcf().number + 1
    
    f = plt.figure(ptn)
    ax = plt.subplot(111)

    plt.plot(np.real(term_rhs1), y, 'b', linewidth=1.5, label="Production")
    plt.plot(np.real(term_rhs2), y, 'm', linewidth=1.5, label=r"$\dfrac{1}{Re Sc} \nabla^2 (\rho'\rho')$")
    plt.plot(np.real(-term_rhs3), y, 'c', linewidth=1.5, label=r"$\tilde{\epsilon}_b = \dfrac{1}{Re Sc}2\nabla\rho'\cdot \nabla\rho'$")
    
    plt.plot(np.real(RHS_dscEq), y, 'r', linewidth=1.5, label="RHS")
    plt.plot(np.real(LHS_dscEq), y, 'k--', linewidth=1.5, label=r"LHS")

    plt.xlabel("density-self-correlation equation balance BOUSSINESQ (NON-DIMENSIONAL)", fontsize=14)
    plt.ylabel('y', fontsize=16)
    plt.gcf().subplots_adjust(left=0.16)
    plt.gcf().subplots_adjust(bottom=0.15)
    plt.ylim([ymin, ymax])

    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=4, fancybox=True, shadow=True, fontsize=10)

    f.show()



    

    #print("LHS_dscEq*rhoref**2*Uref/Lref = ", LHS_dscEq*rhoref**2*Uref/Lref)
    
    # Dimensional growth rate
    omega_i_dim = omega_i*Uref/Lref

    # Make eigenfunctions dimensional
    ueig = ueig*Uref
    veig = veig*Uref
    weig = weig*Uref

    peig = peig*rhoref*Uref**2.
    reig = reig*rhoref

    # Compute dimensional baseflow
    Rho_dim = Rho*rhoref
    Rhop_dim = Rhop*rhoref/Lref

    Rho2_dim = np.multiply(Rho_dim, Rho_dim)

    Rho_dim_inv = 1/Rho_dim
    Rho2_dim_inv = 1/Rho2_dim

    Mu_dim = Mu*muref

    # Compute velocity diveregence
    div_dim = div*Uref/Lref

    ######################################
    # dsc-Equation balance (DIMENSIONAL)
    ######################################

    LHS_dscEq_d = 2*omega_i_dim*compute_inner_prod(reig, reig)

    #print("")
    #print("LHS_dscEq_d = ", LHS_dscEq_d)

    # Note: not taking the triple correlation
    ia_dim = ia/Lref
    ia2_dim = ia2/Lref**2.

    ib_dim = ib/Lref
    ib2_dim = ib2/Lref**2.

    D1_dim = 1/Lref*map.D1
    D2_dim = 1/Lref**2.*map.D2

    Dr_dim = np.matmul(D1_dim, reig)
    D2r_dim = np.matmul(D2_dim, reig)

    term_rhs1 = -2.*np.multiply(Rhop_dim, compute_inner_prod(reig, weig))
    term_rhs2 = 2.*( compute_inner_prod(ia_dim*reig, ia_dim*reig) + compute_inner_prod(ib_dim*reig, ib_dim*reig) + compute_inner_prod(Dr_dim, Dr_dim) )
    term_rhs2 = term_rhs2 + 2.*( compute_inner_prod(reig, ia2_dim*reig) + compute_inner_prod(reig, ib2_dim*reig) + compute_inner_prod(reig, D2r_dim) )
    
    term_rhs3 = -2.*( compute_inner_prod(ia_dim*reig, ia_dim*reig) + compute_inner_prod(ib_dim*reig, ib_dim*reig) + compute_inner_prod(Dr_dim, Dr_dim) )

    # For dimensional equation, we do not have 1/(Re*Sc) but Dref instead
    RHS_dscEq_d = term_rhs1 + Dref*( term_rhs2 + term_rhs3 )


    energy_bal_dscEq_d  = np.amax(np.abs(LHS_dscEq_d-RHS_dscEq_d))
    energy_bal_dscEq222_d  = np.amax(np.abs(LHS_dscEq_d[1:-2]-RHS_dscEq_d[1:-2]))
    print("Energy balance (RT) dsc-equation BOUSSINESQ (dimensional)      = ", energy_bal_dscEq_d)
    print("Energy balance (RT) dsc-equation 222 BOUSSINESQ (dimensional)   = ", energy_bal_dscEq222_d)


    show_now = 0

    if show_now==1:
    
        ptn = ptn + 1
        
        f = plt.figure(ptn)
        ax = plt.subplot(111)
        
        plt.plot(np.real(LHS_dscEq_d), y, 'r', linewidth=1.5, label=r"LHS (dimensional)")
        plt.plot(np.real(RHS_dscEq_d), y, 'k--', linewidth=1.5, label=r"RHS (dimensional)")
        
        plt.xlabel("density-self-correlation equation balance (DIMENSIONAL)", fontsize=14)
        plt.ylabel('y', fontsize=16)
        plt.gcf().subplots_adjust(left=0.16)
        plt.gcf().subplots_adjust(bottom=0.15)
        plt.ylim([ymin, ymax])
        
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=4, fancybox=True, shadow=True, fontsize=10)
        
        f.show()

    input("After density-self-correlation equation balance checks")



def compute_b_eq_ener_bal_boussi(ueig, veig, weig, peig, reig, D1, D2, y, alpha, beta, omega_i, bsfl, bsfl_ref, mob, rt_flag):

    ny = len(ueig)
    #plt.close('all')

    #print("omega_i = ", omega_i)
    
    # Get some quantities to compute the balances
    Mu, Mup, Rho, Rhop, Rhopp, rho2, rho3, rho_inv, rho2_inv, rho3_inv = get_baseflow_and_derivatives(bsfl, mob)
    ia, ib, ia2, ib2, iab, Du, Dv, Dw, Dr, D2u, D2v, D2w, D2r = get_eigenfunctions_quantities(alpha, beta, D1, D2, ueig, veig, weig, reig)
    
    Sc = bsfl_ref.Sc
    Re = bsfl_ref.Re
    Fr = bsfl_ref.Fr

    Lref = bsfl_ref.Lref
    Uref = bsfl_ref.Uref
    muref = bsfl_ref.muref
    rhoref = bsfl_ref.rhoref

    #
    # Compute non-dimensional b-equation balance
    #

    # velocity induced for Boussinesq LST
    vel_boussi_lst_exact = 1/(Re*Sc)*(-2.*y/bsfl.delta**2.)
    #vel_boussi_lst_exact = 1/(Re*Sc)*np.divide(Rhopp, Rhop) ==> this gives nan's

    v = -np.divide(reig, rho2)

    dvdx = ia*v
    dvdy = ib*v
    dvdz = np.matmul(D1, v)

    uvel_dvdx = compute_inner_prod(ueig, dvdx)
    vvel_dvdy = compute_inner_prod(veig, dvdy)
    wvel_dvdz = compute_inner_prod(weig, dvdz)
    
    div = ( 1j*alpha*ueig + 1j*beta*veig + Dw )

    # Left-hand-side
    LHS_bEq = 2*omega_i*np.multiply(rho2_inv, compute_inner_prod(reig, reig))

    # Right-hand-side
    prod_bsfl = 2.*np.multiply(rho2_inv, Rhop)
    prod_dist = -np.multiply(prod_bsfl, compute_inner_prod(reig, weig))

    # Compute Laplacian of b (see leaflet on Laplacian of b)
    Laplacian_b = 2.*np.multiply( rho2_inv, compute_inner_prod(ia*reig, ia*reig) + \
                                            compute_inner_prod(ib*reig, ib*reig) + \
                                            compute_inner_prod(Dr, Dr) + \
                                            compute_inner_prod(reig, ia2*reig) + \
                                            compute_inner_prod(reig, ib2*reig) + \
                                            compute_inner_prod(reig, D2r) )

    rho4_inv = np.multiply(rho2_inv, rho2_inv)
    
    lap1_bsfl = -8.*np.multiply(rho3_inv, Rhop)
    lap2_bsfl = 6.*np.multiply(rho4_inv, np.multiply(Rhop, Rhop) )
    lap3_bsfl = -2.*np.multiply(rho3_inv, Rhopp)
    
    Laplacian_b = Laplacian_b + np.multiply(lap1_bsfl, compute_inner_prod(reig, Dr)) \
                              + np.multiply(lap2_bsfl, compute_inner_prod(reig, reig)) \
                              + np.multiply(lap3_bsfl, compute_inner_prod(reig, reig))

    Laplacian_b = 1/(Re*Sc)*Laplacian_b

    visc_other1_bsfl = -6.*np.multiply(rho4_inv, np.multiply(Rhop, Rhop) )
    visc_other1_dist = np.multiply(visc_other1_bsfl, compute_inner_prod(reig, reig))
    visc_other1_dist = 1/(Re*Sc)*visc_other1_dist

    visc_other2_bsfl = np.multiply(8.*rho3_inv, Rhop)
    visc_other2_dist = np.multiply(visc_other2_bsfl, compute_inner_prod(reig, Dr) )
    visc_other2_dist = 1/(Re*Sc)*visc_other2_dist

    visc_other3_bsfl = 2.*np.multiply(rho3_inv, Rhopp)
    visc_other3_dist = np.multiply(visc_other3_bsfl, compute_inner_prod(reig, reig) )
    visc_other3_dist = 1/(Re*Sc)*visc_other3_dist

    epsilon_tilde = 2.*np.multiply( rho2_inv, compute_inner_prod(ia*reig, ia*reig) + \
                                            compute_inner_prod(ib*reig, ib*reig) + \
                                            compute_inner_prod(Dr, Dr) )

    epsilon_b_tilde = 1/(Re*Sc)*epsilon_tilde

    RHS_bEq = prod_dist + Laplacian_b + visc_other1_dist + visc_other2_dist + visc_other3_dist - epsilon_b_tilde

    # Note: term ui_bsfl*db/dxi ==> gives two extra term, even for Boussinesq!!!!!!
    conv_term1_bsfl = 2.*np.multiply(rho3_inv, Rhopp)
    conv_term1_dist = 1/(Re*Sc)*np.multiply(conv_term1_bsfl, compute_inner_prod(reig, reig))
    
    conv_term2_bsfl = np.divide(vel_boussi_lst_exact, rho2)
    conv_term2_dist = -2*np.multiply(conv_term2_bsfl, compute_inner_prod(reig, Dr))

    # If I add conv_term2_dist, balance is currently not satisfied. Maybe because I would need to add it to stability Equation???
    #RHS_bEq = RHS_bEq + conv_term1_dist #+ conv_term2_dist
        
    energy_bal_bEq  = np.amax(np.abs(LHS_bEq-RHS_bEq))
    print("Energy balance (RT) b-equation NONDIM (BOUSSINESQ)       = ", energy_bal_bEq)
    #print("")


    ptn = plt.gcf().number + 1

    ymin, ymax = FindResonableYminYmax(epsilon_b_tilde, y)

    f = plt.figure(ptn)
    ax = plt.subplot(111)
    
    plt.plot(np.real(prod_dist), y, 'k--', linewidth=1.5, label=r"Production")
    plt.plot(np.real(Laplacian_b), y, 'b--', linewidth=1.5, label=r"$\dfrac{1}{Re Sc}\nabla^2 b$")
    
    plt.plot(np.real(visc_other1_dist), y, 'r--', linewidth=1.5, label=r"$-\dfrac{6}{Re Sc} \dfrac{b}{\bar{\rho}^2}\dfrac{\partial \bar{\rho}}{\partial z}\dfrac{\partial \bar{\rho}}{\partial z}$")
    plt.plot(np.real(visc_other2_dist), y, 'g', linewidth=1.5, label=r"$\dfrac{8}{\bar{\rho}^3}\dfrac{\partial \bar{\rho}}{\partial z}\overline{\rho' \dfrac{\partial \rho'}{\partial z}}$")
    plt.plot(np.real(visc_other3_dist), y, 'm--', linewidth=1.5, label=r"$\dfrac{2b}{\bar{\rho}}\dfrac{\partial^2 \bar{\rho}}{\partial z^2}$")
    plt.plot(np.real(epsilon_b_tilde), y, 'c--', linewidth=1.5, label=r"$\tilde{\epsilon}_b$")
    plt.plot(np.real(RHS_bEq), y, 'k', linewidth=1.5, label=r"RHS")
    plt.plot(np.real(LHS_bEq), y, 'g--', linewidth=1.5, label=r"LHS")
    
    plt.xlabel("b-equation balance BOUSSINESQ (NON-DIMENSIONAL)", fontsize=14)
    plt.ylabel('y', fontsize=16)
    plt.gcf().subplots_adjust(left=0.16)
    plt.gcf().subplots_adjust(bottom=0.15)
    plt.ylim([ymin, ymax])
    
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=4, fancybox=True, shadow=True, fontsize=10)
    
    f.show()


    #######################################################
    #######################################################

    ptn = plt.gcf().number + 1

    f = plt.figure(ptn)
    ax = plt.subplot(111)
    
    plt.plot(np.real(uvel_dvdx), y, 'k', linewidth=1.5, label=r"$\overline{u'\dfrac{\partial v}{\partial x}}$")
    plt.plot(np.real(wvel_dvdz), y, 'r', linewidth=1.5, label=r"$\overline{w'\dfrac{\partial v}{\partial z}}$")
        
    plt.xlabel("b-equation balance BOUSSINESQ (NON-DIMENSIONAL) ==> MORE TERMS", fontsize=14)
    plt.ylabel('y', fontsize=16)
    plt.gcf().subplots_adjust(left=0.16)
    plt.gcf().subplots_adjust(bottom=0.15)
    plt.ylim([ymin, ymax])
    
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=4, fancybox=True, shadow=True, fontsize=10)
    
    f.show()

    input("End routine for b-equation")



def compute_a_eq_ener_bal_boussi_Sc_1(ueig, veig, weig, peig, reig, D1, D2, y, alpha, beta, omega_i, bsfl, bsfl_ref, mob, rt_flag, KE_dist, epsilon_dist, epsil_tild_min_2tild_dist, iarr):

    #plt.close('all')

    ny = len(ueig)
    
    # Get some quantities to compute the balances
    Mu, Mup, Rho, Rhop, Rhopp, rho2, rho3, rho_inv, rho2_inv, rho3_inv = get_baseflow_and_derivatives(bsfl, mob)
    ia, ib, ia2, ib2, iab, Du, Dv, Dw, Dr, D2u, D2v, D2w, D2r = get_eigenfunctions_quantities(alpha, beta, D1, D2, ueig, veig, weig, reig)
    
    Sc = bsfl_ref.Sc
    Re = bsfl_ref.Re
    Fr = bsfl_ref.Fr

    Lref = bsfl_ref.Lref
    Uref = bsfl_ref.Uref
    muref = bsfl_ref.muref
    rhoref = bsfl_ref.rhoref

    div = ( 1j*alpha*ueig + 1j*beta*veig + Dw )

    v = -np.divide(reig, rho2)

    dpdx = (1j*alpha)*peig
    dpdy = (1j*beta)*peig
    dpdz = np.matmul(D1, peig) 
    Dp  = dpdz

    # Specific volume derivatives
    dvdx = (1j*alpha)*v
    dvdy = (1j*beta)*v
    dvdz = np.matmul(D1, v)

    
    dwdx    = (1j*alpha)*weig
    dwdy    = (1j*beta)*weig
    dwdz    = Dw

    drdx    = (1j*alpha)*reig
    drdy    = (1j*beta)*reig
    drdz    = Dr

    d2wdx2  = (1j*alpha)**2.*weig
    d2wdy2  = (1j*beta)**2.*weig
    d2wdz2  = D2w

    d2rdx2  = (1j*alpha)**2.*reig
    d2rdy2  = (1j*beta)**2.*reig
    d2rdz2  = D2r

    # LHS for a-equation
    LHS_aEq = 2*omega_i*np.divide(compute_inner_prod(reig, weig), Rho)
    
    # Compute Laplacian of a
    Lap_bsfl1 = -2.*np.multiply(rho2_inv, Rhop)
    Lap_t1 = np.multiply(Lap_bsfl1, compute_inner_prod(reig, Dw))
    Lap_t2 = np.multiply(Lap_bsfl1, compute_inner_prod(weig, Dr))

    Lap_t3 = np.multiply(rho_inv, compute_inner_prod(reig, d2wdx2 + d2wdy2 + d2wdz2))
    Lap_t4 = np.multiply(rho_inv, compute_inner_prod(weig, d2rdx2 + d2rdy2 + d2rdz2))

    Lap_t5 = 2.*np.multiply(rho_inv, compute_inner_prod(drdx, dwdx) + compute_inner_prod(drdy, dwdy), compute_inner_prod(Dr, Dw))

    Lap_bsfl6 = 2.*np.multiply(rho3_inv, np.multiply(Rhop, Rhop))
    Lap_t6 = np.multiply(Lap_bsfl6, compute_inner_prod(reig, weig))

    Lap_bsfl7 = -np.multiply(rho2_inv, Rhopp)
    Lap_t7 = np.multiply(Lap_bsfl7, compute_inner_prod(reig, weig))

    # RHS: inviscid terms
    RHS_a_t1 = np.multiply(Rho, compute_inner_prod(v, Dp) )
    RHS_a_t2 = -np.multiply(rho_inv, np.multiply(Rhop, compute_inner_prod(weig, weig)))

    RHS_a_t8 = -np.multiply(rho_inv, compute_inner_prod(reig, reig))/Fr**2.

    # RHS: Viscous terms
    Lap_a = Lap_t1 + Lap_t2 + Lap_t3 + Lap_t4 + Lap_t5 + Lap_t6 + Lap_t7
    Lap_a = Lap_a/Re

    RHS_a_bsfl3 = 2.*np.multiply(rho2_inv, Rhop)
    RHS_a_t3 = np.multiply(RHS_a_bsfl3, compute_inner_prod(reig, Dw) + compute_inner_prod(weig, Dr))
    RHS_a_t3 = RHS_a_t3/Re

    RHS_a_bsfl33 = -2.*np.multiply(rho3_inv, np.multiply(Rhop, Rhop))
    RHS_a_t33 = np.multiply(RHS_a_bsfl33, compute_inner_prod(reig, weig))
    RHS_a_t33 = RHS_a_t33/Re

    RHS_a_bsfl4 = np.multiply(rho2_inv, Rhopp)
    RHS_a_t4 = np.multiply(RHS_a_bsfl4, compute_inner_prod(reig, weig))
    RHS_a_t4 = RHS_a_t4/Re

    RHS_a_bsfl5 = -2.*np.multiply(rho_inv, 1.)
    RHS_a_t5 = np.multiply(RHS_a_bsfl5, compute_inner_prod(drdx, dwdx) + compute_inner_prod(drdy, dwdy), compute_inner_prod(Dr, Dw) )
    RHS_a_t5 = RHS_a_t5/Re

    RHS_visc = Lap_a + RHS_a_t3 + RHS_a_t33 + RHS_a_t4 + RHS_a_t5

    RHS_aEq  = RHS_a_t1 + RHS_a_t2 + RHS_a_t8 + RHS_visc

    energy_bal_aEq  = np.amax(np.abs(LHS_aEq-RHS_aEq))
    print("Energy balance (RT) a-equation (BOUSSINESQ ----> ONLY FOR Sc=1):", energy_bal_aEq)

    ####################
    # Model Terms
    ####################
    dpdz_bsfl = -Rho/Fr**2*1
    
    az = np.divide(compute_inner_prod(reig, weig), Rho)
    b = np.divide(compute_inner_prod(reig, reig), rho2 )
    dbdz = np.matmul(D1, b)

    l_taylor_nondim = iarr.l_taylor_dim/Lref
    l_taylor_nondim2 = np.multiply(l_taylor_nondim, l_taylor_nondim)

    pres_dissipation_model = 1./3.*np.multiply(b, dpdz_bsfl)
    pres_diffusion_model = -np.matmul(D1, np.multiply(l_taylor_nondim2, dbdz))
    
    ####################
    # PLOTS
    ####################

    # GET ymin-ymax
    ymin, ymax = FindResonableYminYmax(RHS_aEq, y)

    ptn = plt.gcf().number + 1
    
    f = plt.figure(ptn)
    ax = plt.subplot(111)

    plt.plot(np.real(RHS_a_t1), y, 'b--', linewidth=1.5, label=r"$\bar{\rho} \overline{ v' \dfrac{\partial p'}{\partial z} }$")
    plt.plot(np.real(RHS_a_t2), y, 'g', linewidth=1.5, label=r"$-\dfrac{1}{\bar{\rho}} \dfrac{\partial \bar{\rho}}{\partial z} \overline{w'w'}$")
    plt.plot(np.real(RHS_a_t8), y, 'b', linewidth=1.5, label=r"$-\bar{\rho}g b/Fr^2$")    
    plt.plot(np.real(Lap_a), y, 'm', linewidth=1.5, label=r"$\dfrac{1}{Re}\nabla^2 a_z$")    
    plt.plot(np.real(RHS_a_t3+RHS_a_t33), y, 'c', linewidth=1.5, label=r"$\dfrac{1}{Re} \dfrac{2}{\bar{\rho}}\dfrac{\partial \bar{\rho}}{\partial z} \dfrac{\partial a_z}{\partial z}$")
    plt.plot(np.real(RHS_a_t4), y, 'b-.', linewidth=1.5, label=r"$\dfrac{1}{Re}\dfrac{a_z}{\bar{\rho}}\dfrac{\partial^2 \bar{\rho}}{\partial z^2}$")
    plt.plot(np.real(RHS_a_t5), y, 'm-.', linewidth=1.5, label=r"$-\dfrac{1}{Re}\dfrac{2}{\bar{\rho}} \overline{\dfrac{\partial \rho'}{\partial x_i}\dfrac{\partial w'}{\partial x_i}}$")
    
    plt.plot(np.real(RHS_aEq), y, 'r', linewidth=1.5, label="RHS")
    plt.plot(np.real(LHS_aEq), y, 'k--', linewidth=1.5, label=r"LHS: $\dfrac{\partial a_z}{\partial t}$")

    plt.plot(np.real(pres_diffusion_model), y, 'c--', linewidth=1.5, label="pressure diffusion model")
    plt.plot(np.real(pres_dissipation_model), y, 'c-.', linewidth=1.5, label="pressure dissipation model")
    plt.plot(np.real(pres_dissipation_model+pres_diffusion_model), y, 'c:', linewidth=1.5, label="pressure dissipation + pressure diffusion model")
    
    plt.xlabel(r"$a_z$-equation terms (Boussinesq, Sc=1), with $a_z = \dfrac{\overline{\rho' w'}}{\bar{\rho}}$", fontsize=14)
    plt.ylabel('y', fontsize=16)
    plt.gcf().subplots_adjust(left=0.16)
    plt.gcf().subplots_adjust(bottom=0.15)
    plt.ylim([ymin, ymax])

    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=4, fancybox=True, shadow=True, fontsize=10)

    f.show()
    
    # Compare model terms to exact ones

    Ca1 = 3.2

    #print("KE_dist**0.5", KE_dist**0.5)
    #print("np.sqrt(KE_dist) = ", np.sqrt(KE_dist))
    
    Sturb = np.divide(KE_dist**1.5, epsilon_dist)

    # Compute Ca1_lst
    vdpdz = compute_inner_prod(v, Dp)
    
    Ca1_t1 = np.divide(Sturb, np.multiply(KE_dist**0.5, az))
    
    Ca1_lst = -np.multiply(vdpdz, Ca1_t1)
    
    term_mod1 = np.multiply(np.divide(np.sqrt(KE_dist), Sturb), az)
    
    MODEL_RHS_a_t1 = -Ca1*np.multiply(Rho, term_mod1)

    #MODEL_RHS_a_t1 = -np.multiply(Ca1_lst, np.multiply(Rho, term_mod1)) # --> thi one should match exactly RHS_a_t1

    # Trying same thing but using epsilon_b instead of epsilon
    #epsilon_b_tilde = 1/(Re*Sc)*( 2.*np.multiply( rho2_inv, compute_inner_prod(ia*reig, ia*reig) + compute_inner_prod(ib*reig, ib*reig) + compute_inner_prod(Dr, Dr) ) )
    #Sturb_aha = np.divide(KE_dist**1.5, epsilon_b_tilde)
    #Ca1_t1_aha = np.divide(Sturb_aha, np.multiply(KE_dist**0.5, az))
    #term_mod1_aha = np.multiply(np.divide(np.sqrt(KE_dist), Sturb_aha), az)
    #MODEL_RHS_a_t1_aha = -Ca1*np.multiply(Rho, term_mod1_aha)

    # Extra term (from Daniel)
    rho4_inv = np.multiply(rho2_inv, rho2_inv)

    rho_bsfl_dpvdx = compute_inner_prod(peig, dvdx) + compute_inner_prod(dpdx, v)
    rho_bsfl_dpvdx = np.multiply(Rho, rho_bsfl_dpvdx)
    
    rho_bsfl_dpvdy = compute_inner_prod(peig, dvdy) + compute_inner_prod(dpdy, v)
    rho_bsfl_dpvdy = np.multiply(Rho, rho_bsfl_dpvdy)
    
    rho_bsfl_dpvdz = compute_inner_prod(peig, dvdz) + compute_inner_prod(dpdz, v)
    rho_bsfl_dpvdz = np.multiply(Rho, rho_bsfl_dpvdz)
    
    g_dbdz = 2.*np.multiply(rho2, compute_inner_prod(reig, drdz) ) -2.*np.multiply(np.multiply(Rho, Rhop), compute_inner_prod(reig, reig) ) 
    g_dbdz = np.multiply(g_dbdz, rho4_inv)

    rho_bsfl_g_dbdz = np.multiply(Rho, g_dbdz)

    rho_bsfl_b = np.multiply(Rho, np.divide(compute_inner_prod(reig, reig), rho2 ) )

    max_RHS_a_t1 = np.amax(np.abs(RHS_a_t1))
    max_rho_bsfl_b = np.amax(np.abs(rho_bsfl_b))

    fac_rho_bsfl_b = max_RHS_a_t1/max_rho_bsfl_b

    
    rho_bsfl_b_normalized = rho_bsfl_b/max_rho_bsfl_b
    MODEL_RHS_a_t1_mods_aha = np.multiply(MODEL_RHS_a_t1, rho_bsfl_b_normalized)

    # Plot
    ptn = plt.gcf().number + 1

    #ymin = -0.02
    #ymax = -ymin

    f = plt.figure(ptn)
    ax = plt.subplot(111)

    plt.plot(np.real(RHS_a_t1), y, 'k', linewidth=1.5, label=r"$\bar{\rho} \overline{ v' \dfrac{\partial p'}{\partial z} }$")
    plt.plot(np.real(MODEL_RHS_a_t1), y, 'r--', linewidth=1.5, label=r"$-C_{a1}\bar{\rho} \dfrac{\sqrt{K}}{S}a_z$")
    #plt.plot(np.real(MODEL_RHS_a_t1_mods_aha), y, 'r', linewidth=1.5, label=r"$-C_{a1}\bar{\rho} \dfrac{\sqrt{K}}{S}a_z \times norm(\rho b)$")

    plt.plot(np.real(rho_bsfl_dpvdx), y, 'm', linewidth=1.5, label=r"$\bar{\rho}\dfrac{\partial \left( \overline{p'v'} \right)}{\partial x}$")
    plt.plot(np.real(rho_bsfl_dpvdy), y, 'c--', linewidth=1.5, label=r"$\bar{\rho}\dfrac{\partial \left( \overline{p'v'} \right)}{\partial y}$")
    plt.plot(np.real(rho_bsfl_dpvdz), y, 'g--', linewidth=1.5, label=r"$\bar{\rho}\dfrac{\partial \left( \overline{p'v'} \right)}{\partial z}$")

    plt.plot(np.real(rho_bsfl_g_dbdz), y, 'y--', linewidth=1.5, label=r"$\bar{\rho}  g \dfrac{\partial b}{\partial z}$")

    #plt.plot(np.real(epsil_tild_min_2tild_dist), y, 'b', linewidth=1.5, label=r"$\tilde{\epsilon} - \tilde{\tilde{\epsilon}}$")

    label_rho_bsfl_b = r"$\bar{\rho} b$ ( scaling = %s )" % str('{:.2e}'.format(fac_rho_bsfl_b))    
    plt.plot(np.real(rho_bsfl_b)*fac_rho_bsfl_b, y, 'b-.', linewidth=1.5, label=label_rho_bsfl_b)

    #plt.plot(np.real(MODEL_RHS_a_t1_aha), y, 'b--', linewidth=1.5, label=r"$-C_{a1}\bar{\rho} \dfrac{\sqrt{K}}{S}a_z$")

    plt.xlabel(r"$a_z$-equation terms (Boussinesq, Sc=1), with $a_z = \dfrac{\overline{\rho' w'}}{\bar{\rho}}$", fontsize=14)
    plt.ylabel('y', fontsize=16)
    plt.gcf().subplots_adjust(left=0.16)
    plt.gcf().subplots_adjust(bottom=0.15)
    plt.ylim([ymin, ymax])

    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=4, fancybox=True, shadow=True, fontsize=10)

    f.show()


    # Plot Ca1 coefficient computed from LST data
    
    ptn = plt.gcf().number + 1

    #ymin = -0.02
    #ymax = -ymin
    
    f = plt.figure(ptn)
    ax = plt.subplot(111)

    plt.plot(np.real(Ca1_lst), y, 'k', linewidth=1.5, label=r"$C_{a1}=-\overline{v' \dfrac{\partial p'}{\partial z}} \dfrac{S}{\sqrt{K} a_z}$")

    plt.xlabel(r"$a_z$-equation terms (Boussinesq, Sc=1), $C_{a1}$ coefficient", fontsize=14)
    plt.ylabel('y', fontsize=16)
    plt.gcf().subplots_adjust(left=0.16)
    plt.gcf().subplots_adjust(bottom=0.15)
    plt.ylim([ymin, ymax])

    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=4, fancybox=True, shadow=True, fontsize=10)

    f.show()


    # One more plot
    
    pv_dist = compute_inner_prod(peig, v)
    dpv_dist_dz = np.matmul(D1, pv_dist)

    v_dpdz = compute_inner_prod(v, Dp)
    p_dvdz = compute_inner_prod(peig, dvdz)

    check_v_dpdz = dpv_dist_dz - p_dvdz

    ptn = plt.gcf().number + 1

    f = plt.figure(ptn)
    ax = plt.subplot(111)

    plt.plot(np.real(dpv_dist_dz), y, 'k', linewidth=1.5, label=r"$\dfrac{\partial}{\partial z} \overline{p' v'}$")
    plt.plot(np.real(p_dvdz), y, 'r', linewidth=1.5, label=r"$\overline{p' \dfrac{\partial v'}{\partial z}}$")
    plt.plot(np.real(v_dpdz), y, 'b', linewidth=1.5, label=r"$\overline{v' \dfrac{\partial p'}{\partial z}}$")
    plt.plot(np.real(az), y, 'g', linewidth=1.5, label=r"$a_z=\dfrac{\overline{\rho' w'}}{\bar{\rho}}$")
    plt.plot(np.real(check_v_dpdz), y, 'y-.', linewidth=1.5, label=r"$\dfrac{\partial}{\partial z} \overline{p' v'} - \overline{p' \dfrac{\partial v'}{\partial z}}$")

    plt.plot(np.real(MODEL_RHS_a_t1), y, 'c', linewidth=1.5, label=r"$-Ca_1 \bar{\rho} \dfrac{\sqrt{K}}{S} a_z$ (BHR 32, Schwarzkopf 2011 (19))")
    plt.plot(np.real(-0.5*b), y, 'm', linewidth=1.5, label=r"$-0.5*gref*b$ (Daniel, page 11")


    plt.xlabel(r"$a_z$-equation terms (Boussinesq, Sc=1) =====> MORE TERMS", fontsize=14)
    plt.ylabel('y', fontsize=16)
    plt.gcf().subplots_adjust(left=0.16)
    plt.gcf().subplots_adjust(bottom=0.15)
    plt.ylim([ymin, ymax])

    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=4, fancybox=True, shadow=True, fontsize=10)

    f.show()
    
    input("End of compute_a_eq_ener_bal_boussi_Sc_1 subroutine")

    
def compute_ener_bal_specific_volume_divergence_correlation(ueig, veig, weig, peig, reig, D1, D2, y, alpha, beta, omega_i, bsfl, bsfl_ref, mob, rt_flag):

    #plt.close('all')

    ny = len(ueig)
    
    # Get some quantities to compute the balances
    Mu, Mup, Rho, Rhop, Rhopp, rho2, rho3, rho_inv, rho2_inv, rho3_inv = get_baseflow_and_derivatives(bsfl, mob)
    ia, ib, ia2, ib2, iab, Du, Dv, Dw, Dr, D2u, D2v, D2w, D2r = get_eigenfunctions_quantities(alpha, beta, D1, D2, ueig, veig, weig, reig)

    rho4_inv = np.multiply(rho2_inv, rho2_inv)
    
    Sc = bsfl_ref.Sc
    Re = bsfl_ref.Re
    Fr = bsfl_ref.Fr

    Lref = bsfl_ref.Lref
    Uref = bsfl_ref.Uref
    muref = bsfl_ref.muref
    rhoref = bsfl_ref.rhoref

    div = ( 1j*alpha*ueig + 1j*beta*veig + Dw )

    v   = -np.divide(reig, rho2)
    Dp  = np.matmul(D1, peig)

    drdx    = (1j*alpha)*reig
    drdy    = (1j*beta)*reig
    drdz    = Dr

    d2rdx2  = (1j*alpha)**2.*reig
    d2rdy2  = (1j*beta)**2.*reig
    d2rdz2  = D2r

    LHS = 2.*np.multiply(Rho, compute_inner_prod(v, div))

    # Build RHS
    rhs_t1_bsfl = 4.*np.multiply(rho4_inv, np.multiply(Rhop, Rhop))
    rhs_t2_bsfl = -2.*np.multiply(rho3_inv, Rhopp)
    rhs_t3_bsfl = -4.*np.multiply(rho3_inv, Rhop)
    rhs_t4_bsfl = np.multiply(rho2_inv, 1.)

    rhs_t1_dist = 1/(Re*Sc)*np.multiply(rhs_t1_bsfl, compute_inner_prod(reig, reig))
    rhs_t2_dist = 1/(Re*Sc)*np.multiply(rhs_t2_bsfl, compute_inner_prod(reig, reig))
    rhs_t3_dist = 1/(Re*Sc)*np.multiply(rhs_t3_bsfl, compute_inner_prod(reig, Dr))

    # Laplacian(rho'^2)
    Lap_part1 = 2.*( compute_inner_prod(reig, d2rdx2) + compute_inner_prod(reig, d2rdy2) + compute_inner_prod(reig, d2rdz2) )
    Lap_part2 = 2.*( compute_inner_prod(drdx, drdx) + compute_inner_prod(drdy, drdy) + compute_inner_prod(drdz, drdz) )

    Laplacian = Lap_part1 + Lap_part2
    
    rhs_t4_dist = 1/(Re*Sc)*np.multiply(rhs_t4_bsfl, Laplacian)
    rhs_t5_dist = 2/(Re*Sc)*np.multiply(rhs_t4_bsfl, compute_inner_prod(drdx, drdx) + compute_inner_prod(drdy, drdy) + compute_inner_prod(drdz, drdz) )
    
    
    RHS = rhs_t1_dist + rhs_t2_dist + rhs_t3_dist + rhs_t4_dist - rhs_t5_dist


    # Check that quantities are real
    Check_rhs_t1_dist = np.max(np.abs(rhs_t1_dist.imag))
    Check_rhs_t2_dist = np.max(np.abs(rhs_t2_dist.imag))
    Check_rhs_t3_dist = np.max(np.abs(rhs_t3_dist.imag))
    Check_rhs_t4_dist = np.max(np.abs(rhs_t4_dist.imag))
    Check_rhs_t5_dist = np.max(np.abs(rhs_t5_dist.imag))

    Check_LHS = np.max(np.abs(LHS.imag))
    Check_RHS = np.max(np.abs(RHS.imag))

    tol = 1e-20
    
    if ( Check_rhs_t1_dist < tol and Check_rhs_t2_dist < tol and Check_rhs_t3_dist < tol and \
         Check_rhs_t4_dist < tol and Check_rhs_t5_dist < tol and Check_LHS < tol and Check_RHS < tol ):
        rhs_t1_dist = rhs_t1_dist.real
        rhs_t2_dist = rhs_t2_dist.real
        rhs_t3_dist = rhs_t3_dist.real
        rhs_t4_dist = rhs_t4_dist.real
        rhs_t5_dist = rhs_t5_dist.real
        
        LHS = LHS.real
        RHS = RHS.real
        
    else:
        sys.exit("Error: some quantities have a nonzero imaginary part -- specific-volume-divergence correlation")

    #
    # PLOT
    #

    # Get ymin-ymax
    ymin, ymax = FindResonableYminYmax(RHS, y)
        
    ptn = plt.gcf().number + 1
    
    f  = plt.figure(ptn)
    ax = plt.subplot(111)
    ax.plot(rhs_t1_dist, y, 'b', linewidth=1.5, label=r"$\dfrac{1}{Pe}\dfrac{4}{\bar{\rho}^4} \dfrac{\partial \bar{\rho}}{\partial x_n}\dfrac{\partial \bar{\rho}}{\partial x_n} \overline{\rho'^2}$")
    ax.plot(rhs_t2_dist, y, 'g', linewidth=1.5, label=r"$-\dfrac{1}{Pe}\dfrac{2}{\bar{\rho}^3} \dfrac{\partial^2 \bar{\rho}}{\partial x_n \partial x_n} \overline{\rho'^2}$")
    ax.plot(rhs_t3_dist, y, 'm', linewidth=1.5, label=r"$-\dfrac{1}{Pe}\dfrac{4}{\bar{\rho}^3} \dfrac{\partial \bar{\rho}}{\partial x_n} \overline{\rho' \dfrac{\partial \rho'}{\partial x_n}}$")
    #ax.plot(rhs_t4_dist, y, 'c', linewidth=1.5, label=r"$\dfrac{2}{\bar{\rho}^2} \overline{\rho'\dfrac{\partial^2 \rho'}{\partial x_n \partial x_n}}$")
    ax.plot(rhs_t4_dist, y, 'c', linewidth=1.5, label=r"$\dfrac{1}{Pe}\dfrac{1}{\bar{\rho}^2} \nabla^2\overline{\rho'^2}$")
    ax.plot(rhs_t5_dist, y, 'y', linewidth=1.5, label=r"$\dfrac{1}{Pe}\dfrac{2}{\bar{\rho}^2} \overline{\dfrac{\partial \rho'}{\partial x_n} \dfrac{\partial \rho'}{\partial x_n}}$")
    
    ax.plot(LHS, y, 'k', linewidth=1.5, label=r"$LHS = 2\bar{\rho}\overline{v'\dfrac{\partial u'_j}{\partial x_j}}$")
    ax.plot(RHS, y, 'r-.', linewidth=1.5, label=r"RHS")

    plt.xlabel(r"Specific-volume-divergence correlation", fontsize=14)
    plt.ylabel("z", fontsize=18)
    plt.gcf().subplots_adjust(left=0.17)
    plt.gcf().subplots_adjust(bottom=0.15)
    plt.ylim([ymin, ymax])
    
    plt.legend(loc="upper right")
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15),   # box with lower left corner at (x, y)
              ncol=4, fancybox=True, shadow=True, fontsize=10)
    
    f.show()

    input("Plot for specific-volume-divergence correlation")



def IntegrateDistDensity_CompareToPressure(peig, reig, D1, D2, y, alpha, bsfl_ref):

    ny = len(peig)
    
    Fr = bsfl_ref.Fr
    gref = bsfl_ref.gref
    g_nd = gref/gref

    #density_int_real = trapezoid_integration_cum(np.real(reig), y)
    #density_int_imag = trapezoid_integration_cum(np.imag(reig), y)

    #density_int = density_int_real + 1j*density_int_imag
    #pres_comp = -g_nd*density_int/Fr**2.

    #abs_pres_comp = np.abs(pres_comp)

    #C = pres_comp[-1]/y[-1]
    #print("C=", C)

    #pres_comp = pres_comp + C*y

    # Solve Linear System
    D2_with_bc = D2

    D2_with_bc[0,:] = 0.0
    D2_with_bc[0,0] = 1.0
    
    D2_with_bc[-1,:] = 0.0
    D2_with_bc[-1,-1] = 1.0

    # Add diagonal elements
    for i in range(0, ny):
        D2_with_bc[i,i] = D2_with_bc[i,i] -alpha**2.

    vec_rhs = -g_nd/Fr**2*np.matmul(D1, reig)
    vec_rhs[0] = 0.
    vec_rhs[-1] = 0.
    
    SOL = linalg.solve(D2_with_bc, vec_rhs)

    #idx = int((ny+1)/2)
    #val_max_p = np.amax(np.abs(peig))

    #val_max_SOL = np.amax(np.abs(SOL))
    #SOL = SOL/val_max_SOL*val_max_p

    # PLOT
    ############
    
    ymin = -0.08
    ymax = -ymin
    
    ptn = plt.gcf().number + 1
    
    f  = plt.figure(ptn)
    ax = plt.subplot(111)
    ax.plot(np.real(peig), y, 'b', linewidth=1.5, label=r"pressure")
    #ax.plot(np.real(pres_comp), y, 'r--', linewidth=1.5, label=r"pressure by integration")
    ax.plot(np.real(SOL), y, 'k--', linewidth=1.5, label=r"pressure: solution Poisson eq.")

    plt.xlabel(r"Pressure", fontsize=14)
    plt.ylabel("z", fontsize=18)
    plt.gcf().subplots_adjust(left=0.17)
    plt.gcf().subplots_adjust(bottom=0.15)
    plt.ylim([ymin, ymax])
    
    plt.legend(loc="upper right")
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15),   # box with lower left corner at (x, y)
              ncol=2, fancybox=True, shadow=True, fontsize=10)
    
    f.show()


    # Plot other quantities
    d2pdx2 = (1j*alpha)**2.*peig
    d2pdz2 = np.matmul(D2, peig)
    
    # ptn = plt.gcf().number + 1
    
    # f  = plt.figure(ptn)
    # ax = plt.subplot(111)
    # ax.plot(np.real(d2pdx2), y, 'b', linewidth=1.5, label=r"real(d2pdx2)")
    # ax.plot(np.real(d2pdz2), y, 'r', linewidth=1.5, label=r"real(d2pdz2)")

    # ax.plot(np.imag(d2pdx2), y, 'k', linewidth=1.5, label=r"imag(d2pdx2)")
    # ax.plot(np.imag(d2pdz2), y, 'g', linewidth=1.5, label=r"imag(d2pdz2)")

    # plt.xlabel(r"d2pdx2 and d2pdz2", fontsize=14)
    # plt.ylabel("z", fontsize=18)
    # plt.gcf().subplots_adjust(left=0.17)
    # plt.gcf().subplots_adjust(bottom=0.15)
    # plt.ylim([ymin, ymax])
    
    # plt.legend(loc="upper right")
    # ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15),   # box with lower left corner at (x, y)
    #           ncol=2, fancybox=True, shadow=True, fontsize=10)
    
    # f.show()

    # Check that d2p'/dx2 + d2p'/dz2 = -g/Fr^2*dr'/dz

    
    #print("gref, Fr = ", gref, Fr)

    drdz = np.matmul(D1, reig)
    var_rhs = -g_nd*drdz/Fr**2.

    ptn = plt.gcf().number + 1
    
    f  = plt.figure(ptn)
    ax = plt.subplot(111)
    ax.plot(np.real(d2pdx2+d2pdz2), y, 'b', linewidth=1.5, label=r"real(d2pdx2+d2pdz2)")
    ax.plot(np.real(d2pdx2), y, 'k', linewidth=1.5, label=r"real(d2pdx2)")
    ax.plot(np.real(d2pdz2), y, 'g', linewidth=1.5, label=r"real(d2pdz2)")
    ax.plot(np.real(var_rhs), y, 'r--', linewidth=1.5, label=r"real(var_rhs)")

    plt.xlabel(r"d2pdx2+d2pdz2 and var_rhs", fontsize=14)
    plt.ylabel("z", fontsize=18)
    plt.gcf().subplots_adjust(left=0.17)
    plt.gcf().subplots_adjust(bottom=0.15)
    plt.ylim([ymin, ymax])
    
    plt.legend(loc="upper right")
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15),   # box with lower left corner at (x, y)
              ncol=2, fancybox=True, shadow=True, fontsize=10)
    
    f.show()
    
    input("Compare actual pressure and pressure obtained by integrating density")
    



def write_baseflow_out(ynondim, ydim, Rho, Rhop, Rhopp, W, Wp, Rho_dim, Rhop_dim, Rhopp_dim, W_dim, Wp_dim):

    ny = len(ynondim)
    
    # Rho, Rhop, Rhopp,  W and Wp are non-dimensional
    
    data_out = np.column_stack([ ynondim, ydim, Rho, Rhop, Rhopp, W, Wp, Rho_dim, Rhop_dim, Rhopp_dim, W_dim, Wp_dim ])
    datafile_path = "./Baseflow_RT_" + str(ny) + "_pts.dat"

    np.savetxt(datafile_path , data_out, fmt=['%21.11e','%21.11e','%21.11e','%21.11e','%21.11e','%21.11e',\
                                              '%21.11e','%21.11e','%21.11e','%21.11e','%21.11e','%21.11e'])

    print("")
    print("RT Baseflow has been written out to Baseflow_RT.dat")
    print("")


def GetLayerThicknessRT(ny, z, rho_t1, rho_t2, rho):

    # Find zmin
    for i in range(0, ny-1):
        
        if ( rho[i] <= rho_t1 and rho[i+1] > rho_t1 ):
            rho1 = rho[i]
            rho2 = rho[i+1]
            z1 = z[i]
            z2 = z[i+1]
            
            acoef = (rho2-rho1)/(z2-z1)
            bcoef = rho2-acoef*z2

            zmin = (rho_t1-bcoef)/acoef
            print("Thickness of light fluid = ", zmin)
            break

    # Find zmax
    for i in range(0, ny-1):
        
        if ( rho[i] <= rho_t2 and rho[i+1] > rho_t2 ):
            rho1 = rho[i]
            rho2 = rho[i+1]
            z1 = z[i]
            z2 = z[i+1]
            
            acoef = (rho2-rho1)/(z2-z1)
            bcoef = rho2-acoef*z2

            zmax = (rho_t2-bcoef)/acoef
            print("Thickness of heavy fluid = ", zmax)
            #print("zmax = ", zmax)
            break

    H = zmax-zmin

    return H


def GetLayerThicknessRT_CabotCook(ny, z, rho, rho1, rho2):

    # mole fraction
    X = (rho-rho1)/(rho2-rho1)

    Xm = np.zeros(ny, dp)
    
    for i in range(0, ny):
        if   ( X[i] <= 0.5 ):
            #print("here")
            Xm[i] = 2*X[i]
        elif ( X[i] > 0.5 ):
            #print("Hieaouw")
            Xm[i] = 2.*(1.-X[i])
        else:
            sys.exit("error in GetLayerThicknessRT_CabotCook")

    h = trapezoid_integration(Xm, z)

    ptn = plt.gcf().number + 1
    
    f  = plt.figure(ptn)
    ax = plt.subplot(111)
    ax.plot(Xm, z, 'b', linewidth=1.5, label=r"Xm")

    plt.xlabel(r"Xm", fontsize=14)
    plt.ylabel("z", fontsize=18)
    plt.gcf().subplots_adjust(left=0.17)
    plt.gcf().subplots_adjust(bottom=0.15)
    #plt.ylim([ymin, ymax])
    
    #plt.legend(loc="upper right")
    #ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15),   # box with lower left corner at (x, y)
    #          ncol=2, fancybox=True, shadow=True, fontsize=10)
    
    f.show()
    
    input("Xm vs z in GetLayerThicknessRT_CabotCook")

    return h


def GetLayerThicknessRT_Other(ny, z, rho, rho1, rho2):

    fct = np.divide( np.multiply(rho2-rho, rho-rho1), 6.*(rho2-rho1)**2. )
    #fct = rho
    
    h = trapezoid_integration(fct, z)

    #h = h*2/(rho1+rho2)

    return h




def ReScale_NonDim1(bsfl, bsfl_ref, omega_i_dim, alp_dim, bet_dim):

    print("")
    print("Bulk Reynolds number:")
    print("=====================")

    # Uref is here the layer thickness times the growth rate
    Uref_b = bsfl.h_thick*omega_i_dim
    
    #WvelMax = np.max(np.abs(bsfl.Wvel))
    #print("Uref_b, WvelMax = ", Uref_b, WvelMax)
    
    Re_b = bsfl_ref.rhoref*Uref_b*bsfl.h_thick/bsfl_ref.muref
    print("Re_b = ", Re_b)
    print("")

    print("")
    print("Reynolds number alternate:")
    print("==========================")

    # Scaling
    t_alt = 1./np.sqrt(bsfl_ref.At*alp_dim*bsfl_ref.gref)
    l_alt = 1./alp_dim

    # U_alt can be seen as ref. velocity for bubble (see Wilkinson and Jacobs)
    # It can also be seen as time scale based on inviscid growth rate (sqrt(A*k*g)) and length scale based on wavelength of instability
    U_alt = np.sqrt(bsfl_ref.At*bsfl_ref.gref/alp_dim) 

    Re_alt = bsfl_ref.rhoref*U_alt*l_alt/bsfl_ref.muref
    
    print("alpha (dim.)         = ", alp_dim)
    print("U_alt                = ", U_alt)
    print("l_alt/t_alt          = ", l_alt/t_alt)
    print("Inviscid growth rate = ", np.sqrt(bsfl_ref.At*bsfl_ref.gref*alp_dim))
    print("Re_alt               = ", bsfl_ref.rhoref*U_alt*l_alt/bsfl_ref.muref)
    print("U_alt*alp_dim        = ", U_alt*alp_dim)
    print("Uref_b*alp_dim       = ", Uref_b*alp_dim)



def ReScale_NonDim2(bsfl, bsfl_ref, omega_i_dim, alp_dim, bet_dim):

    print("")
    print("Scaling based on thickness layer:")
    print("=================================")

    # Uref
    Uref_o = np.sqrt(bsfl_ref.gref*bsfl.h_thick)
    
    # Scaling
    l_o = bsfl.h_thick
    t_o = np.sqrt(bsfl.h_thick/bsfl_ref.gref)

    tol  = 1.0e-15
    if ( abs(Uref_o-l_o/t_o) > tol ):
        sys.exit("Scaling inconsistency in ReScale_NonDim2")
        
    print("Uref_o, l_o, t_o, l_o/t_o = ", Uref_o, l_o, t_o, l_o/t_o)

    Re_o = bsfl_ref.rhoref*Uref_o*bsfl.h_thick/bsfl_ref.muref
    Fr_o = Uref_o/np.sqrt(bsfl_ref.gref*l_o)
    print("Re_o, Fr_o = ", Re_o, Fr_o)
    print("")

def ReScale_NonDim3(bsfl, bsfl_ref, omega_i_dim, alp_dim, bet_dim):

    print("")
    print("Daniel scaling:")
    print("=================================")

    # Uref
    Uref_o = np.sqrt(bsfl_ref.gref*bsfl.h_thick)
    
    # Scaling
    l_o = bsfl.h_thick
    #t_o = np.sqrt(bsfl.h_thick/bsfl_ref.gref)

    #tol  = 1.0e-15
    #if ( abs(Uref_o-l_o/t_o) > tol ):
    #    sys.exit("Scaling inconsistency in ReScale_NonDim2")
        
    #print("Uref_o, l_o, t_o, l_o/t_o = ", Uref_o, l_o, t_o, l_o/t_o)

    Re_o = bsfl_ref.rhoref*Uref_o*bsfl.h_thick/bsfl_ref.muref
    #Fr_o = Uref_o/np.sqrt(bsfl_ref.gref*l_o)
    print("Re_o (Daniel) = ", Re_o)
    print("")


#def ComputeTaylorScale(bsfl, bsfl_ref, omega_i_dim, alp_dim, bet_dim):

#    nu = np.divide(bsfl.Mu, bsfl.Rho)
    

def compute_disturbance_dissipation_dimensional(ueig, veig, weig, peig, reig, y, alpha, beta, D1, bsfl, bsfl_ref, mob, iarr):

    print("")
    Sc = bsfl_ref.Sc
    Re = bsfl_ref.Re
    Fr = bsfl_ref.Fr

    ny = len(ueig)

    Lref = bsfl_ref.Lref
    Uref = bsfl_ref.Uref
    muref = bsfl_ref.muref
    rhoref = bsfl_ref.rhoref

    visc   = muref/rhoref
    print("visc = ", visc)

    D1_dim = bsfl.D1_dim
    D2_dim = bsfl.D2_dim

    # Make eigenfunctions dimensional
    ueig = ueig*Uref
    veig = veig*Uref
    weig = weig*Uref
    
    peig = peig*rhoref*Uref**2.
    reig = reig*rhoref
    
    # Baseflow
    Rho = bsfl.Rho
    Rhop = bsfl.Rhop
    Rhopp = bsfl.Rhopp

    rho2 = np.multiply(Rho, Rho)
    rho2_inv = 1/rho2
    
    # Dimensional
    ia = 1j*alpha/Lref
    ib = 1j*beta/Lref
    Dr = np.matmul(D1_dim, reig)

    dudx = ia*ueig
    dudy = ib*ueig
    dudz = np.matmul(D1_dim, ueig)
    
    dvdx = ia*veig
    dvdy = ib*veig
    dvdz = np.matmul(D1_dim, veig)
    
    dwdx = ia*weig
    dwdy = ib*weig
    dwdz = np.matmul(D1_dim, weig)
    
    sub1 = dudy+dvdx
    sub2 = dvdz+dwdy
    sub3 = dudz+dwdx

    div = dudx+dvdy+dwdz

    div_non_dim = div*Lref/Uref

    dwdz_nondim = dwdz*Lref/Uref

    print("max(dudx), max(dvdy), max(dwdz) (dimensional) = ", np.max(dudx), np.max(dvdy), np.max(dwdz))
    print("max(dudx), max(dvdy), max(dwdz) (nondimensional) = ", np.max(dudx)*Lref/Uref, np.max(dvdy)*Lref/Uref, np.max(dwdz)*Lref/Uref)
    print("max(div), max(div_non_dim) = ", np.max(div), np.max(div_non_dim))
    print("")
    
    term1 = 2.*( compute_inner_prod(dudx, dudx) + compute_inner_prod(dvdy, dvdy) + compute_inner_prod(dwdz, dwdz) )
    term2 = -2./3.*compute_inner_prod(div, div)
    term3 = compute_inner_prod(sub1, sub1) + compute_inner_prod(sub2, sub2) + compute_inner_prod(sub3, sub3)

    #epsilon_check = visc*( term1 + term2 + term3 )
    
    # Plot diveregence dimensional/non-dimensional
    # --------------------------------------------

    # ptn = plt.gcf().number + 1
    
    # # GET ymin-ymax
    # ymin, ymax = FindResonableYminYmax(div, y)

    # f = plt.figure(ptn)
    # ax = plt.subplot(111)
    # ax.plot(np.abs(div),        y, 'k', linewidth=1.5, label=r"div")
    # ax.plot(np.abs(div_non_dim), y, 'g--', linewidth=1.5, label=r"div_non_dim")

    # plt.xlabel("divergence", fontsize=14)
    # plt.ylabel('z', fontsize=16)
    # plt.gcf().subplots_adjust(left=0.18)
    # plt.gcf().subplots_adjust(bottom=0.15)
    # plt.ylim([ymin, ymax])

    # ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15),   # box with lower left corner at (x, y)
    #           ncol=2, fancybox=True, shadow=True, fontsize=10)
    
    # f.show()

    ########################################
    # Various components of the dissipation
    ########################################

    # 1) visc*(du_i/dx_j)*(du_i/dx_j)
    epsilon_tilde_dist = compute_inner_prod(dudx, dudx) + compute_inner_prod(dudy, dudy) + compute_inner_prod(dudz, dudz) \
                       + compute_inner_prod(dvdx, dvdx) + compute_inner_prod(dvdy, dvdy) + compute_inner_prod(dvdz, dvdz) \
                       + compute_inner_prod(dwdx, dwdx) + compute_inner_prod(dwdy, dwdy) + compute_inner_prod(dwdz, dwdz)
    epsilon_tilde_dist = visc*epsilon_tilde_dist

    # 2) visc*(du_i/dx_j)*(du_j/dx_i)
    epsilon_2tilde_dist = compute_inner_prod(dudx, dudx) + compute_inner_prod(dvdy, dvdy) + compute_inner_prod(dwdz, dwdz) \
                        + 2.*compute_inner_prod(dudy, dvdx) + 2.*compute_inner_prod(dudz, dwdx) + 2.*compute_inner_prod(dvdz, dwdy)
    epsilon_2tilde_dist = visc*epsilon_2tilde_dist

    # 3) incompressible dissipation
    epsilon_incomp_dist = epsilon_tilde_dist + epsilon_2tilde_dist

    # 4) compressible part of dissipation
    epsilon_comp_dist = visc*term2

    # 5) full dissipation
    epsilon_dist = epsilon_incomp_dist + epsilon_comp_dist 


    # Disturbance kinetic energy
    uu = compute_inner_prod(ueig, ueig)
    vv = compute_inner_prod(veig, veig)
    ww = compute_inner_prod(weig, weig)

    print("")
    print("-----------------------> Adding 1e-10 to KE_dist...")
    print("")
        
    KE_dist = 0.5*( uu + vv + ww ) #+ 1.0e-10

    # Check that quantities are real
    Check_eps  = np.max(np.abs(epsilon_dist.imag))
    Check_KE   = np.max(np.abs(KE_dist.imag))

    tol = 1e-20
    
    if ( Check_eps < tol and Check_KE < tol ):
        KE_dist = KE_dist.real
        epsilon_dist = epsilon_dist.real
    else:
        print("np.max(np.abs(epsilon_dist.imag)) = ", np.max(np.abs(epsilon_dist.imag)))
        print("np.max(np.abs(KE_dist.imag))      = ", np.max(np.abs(KE_dist.imag)))
        print("")
        sys.exit("Error: some quantities have a nonzero imaginary part")

    # Specific dissipation omega
    spec_dissip = np.divide(epsilon_dist, KE_dist)

    #
    # Plot KE, Dissipation and Specific dissipation
    #

    # Get ymin-ymax
    ymin, ymax = FindResonableYminYmax(iarr.l_taylor_dim, y)

    ptn = plt.gcf().number + 1

    f = plt.figure(ptn)
    ax = plt.subplot(111)
    ax.plot(KE_dist, y, 'k', linewidth=1.5, label=r"KE")
    ax.plot(epsilon_dist, y, 'r', linewidth=1.5, label=r"$\epsilon$")
    ax.plot(spec_dissip, y, 'c', linewidth=1.5, label=r"Specific dissipation $\omega \approx \epsilon/k$")

    plt.xlabel("Turbulencxe KE and dissipation", fontsize=14)
    plt.ylabel('z', fontsize=16)
    plt.gcf().subplots_adjust(left=0.18)
    plt.gcf().subplots_adjust(bottom=0.15)
    plt.ylim([ymin, ymax])

    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15),   # box with lower left corner at (x, y)
              ncol=2, fancybox=True, shadow=True, fontsize=10)
    
    f.show()

    return KE_dist, epsilon_dist

def compute_reynolds_stresses(ueig, veig, weig, peig, reig, y, bsfl_ref, rt_flag, D1):

    #plt.close('all')

    print("")
    print("Computation of Reynolds stresses")
    print("I did not include the constant term pi/alpha, which will not change the shape of the stresses")

    ny = len(ueig)
        
    Sc = bsfl_ref.Sc
    Re = bsfl_ref.Re
    Fr = bsfl_ref.Fr

    Lref = bsfl_ref.Lref
    Uref = bsfl_ref.Uref
    muref = bsfl_ref.muref
    rhoref = bsfl_ref.rhoref

    # Reynolds stresses
    Rs_uu = compute_inner_prod(ueig, ueig)
    Rs_uv = compute_inner_prod(ueig, veig)
    Rs_uw = compute_inner_prod(ueig, weig)

    Rs_vv = compute_inner_prod(veig, veig)
    Rs_vw = compute_inner_prod(veig, weig)

    Rs_ww = compute_inner_prod(weig, weig)

    d_Rs_uu_dz = np.matmul(D1, Rs_uu)
    d_Rs_ww_dz = np.matmul(D1, Rs_ww)

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

    max_uu = np.amax(np.abs(Rs_uu))
    max_vv = np.amax(np.abs(Rs_vv))
    max_ww = np.amax(np.abs(Rs_ww))

    print("")
    print("max_uu/max_ww = ", max_uu/max_ww)
    print("max_vv/max_ww = ", max_vv/max_ww)

    ######################################
    # PLOT REYNOLDS STRESSES
    ######################################
    
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

def FindResonableYminYmax(var, y):

    ny = len(var)
    
    abs_var = np.abs(var)
    max_var = np.amax(np.abs(var))
    max_var_idx = np.argmax(np.abs(var))

    # Initialize for safety
    idx_beg = 1

    Small = 0.005*max_var

    for i in range(0,ny):
        if (abs_var[i] > Small):
            idx_beg = i
            break

    idx_end = ny-idx_beg

    ymin = y[idx_beg]
    ymax = -ymin

    return ymin, ymax

def close_previous_plots():

    ptn = plt.gcf().number

    #print("ptn = ", ptn)

    for i in range(1, ptn+1):
        plt.close(i)

def reynolds_stresses_balance_equations(ueig, veig, weig, peig, reig, y, mob, bsfl, bsfl_ref, rt_flag, D1, D2, alpha, beta, omega_i, iarr):
    
    ny = len(ueig)
        
    Sc = bsfl_ref.Sc
    Re = bsfl_ref.Re
    Fr = bsfl_ref.Fr

    Lref = bsfl_ref.Lref
    gref = bsfl_ref.gref
    Uref = bsfl_ref.Uref
    muref = bsfl_ref.muref
    rhoref = bsfl_ref.rhoref
    
    ia, ib, ia2, ib2, iab, Du, Dv, Dw, Dr, D2u, D2v, D2w, D2r = get_eigenfunctions_quantities(alpha, beta, D1, D2, ueig, veig, weig, reig)
    Mu, Mup, Rho, Rhop, Rhopp, rho2, rho3, rho_inv, rho2_inv, rho3_inv = get_baseflow_and_derivatives(bsfl, mob)

    dpdx = ia*peig
    dpdy = ib*peig
    dpdz = np.matmul(D1, peig)
    
    # Reynolds stresses
    Rs_uu = compute_inner_prod(ueig, ueig)
    Rs_uv = compute_inner_prod(ueig, veig)
    Rs_uw = compute_inner_prod(ueig, weig)

    Rs_vv = compute_inner_prod(veig, veig)
    Rs_vw = compute_inner_prod(veig, weig)

    Rs_ww = compute_inner_prod(weig, weig)

    # Derivatives
    dudx = ia*ueig
    dudy = ib*ueig
    dudz = Du
    
    dvdx = ia*veig
    dvdy = ib*veig
    dvdz = Dv

    dwdx = ia*weig
    dwdy = ib*weig
    dwdz = Dw
    
    d2udx2  = ia2*ueig
    d2udy2  = ib2*ueig
    d2udz2  = D2u

    d2vdx2  = ia2*veig
    d2vdy2  = ib2*veig
    d2vdz2  = D2v

    d2udxdz = ia*Du
    d2wdxdz = ia*Dw
    
    d2vdxdy = ia*ib*veig
    d2vdydz = ib*Dv

    d2wdx2  = ia2*weig
    d2wdy2  = ib2*weig
    d2wdz2  = D2w

    #print("max(d2udx2 + d2vdxdy + d2wdxdz) = ", np.amax(np.abs(d2udx2 + d2vdxdy + d2wdxdz)))
    
    dTau1k_dxk = 2./Re*d2udx2 + 1/Re*(d2udy2+d2vdxdy) + 1/Re*(d2udz2+d2wdxdz)
    dTau3k_dxk = 2./Re*d2wdz2 + 1/Re*(d2udxdz+d2wdx2) + 1/Re*(d2wdy2+d2vdydz)

    ############################
    # Balance Equation for u'u'
    ############################

    # Should not have rho_bsfl for Boussinesq!!!!!
    LHS_uu = 2*omega_i*Rs_uu
    #LHS_uu = 2*omega_i*np.multiply(Rho, Rs_uu)

    print("max(LHS_uu) = ", np.amax(LHS_uu))
    
    RHS_uu_t1 = -2.*compute_inner_prod(ueig, dpdx)
    # RHS_uu_t2 = 2.*compute_inner_prod(ueig, dTau1k_dxk)

    # # Write RHS_uu_t2 as 2*( d/dx_k(u'tau'_1k) - tau'_1k du'_k/dx_k )
    # ddxk_utau_1k = 2/Re*( compute_inner_prod(dudx, dudx) + compute_inner_prod(ueig, d2udx2) ) +\
    #                1/Re*( compute_inner_prod(dudy, dudy) + compute_inner_prod(dudy, dvdx) + compute_inner_prod(ueig, d2udy2) + compute_inner_prod(ueig, d2vdxdy) ) +\
    #                1/Re*( compute_inner_prod(dudz, dudz) + compute_inner_prod(dudz, dwdx) + compute_inner_prod(ueig, d2udz2) + compute_inner_prod(ueig, d2wdxdz) )

    # tau1k_dudx_k = 2/Re*( compute_inner_prod(dudx, dudx) ) +\
    #                1/Re*( compute_inner_prod(dudy, dudy) + compute_inner_prod(dudy, dvdx) ) +\
    #                1/Re*( compute_inner_prod(dudz, dudz) + compute_inner_prod(dudz, dwdx) )

    # RHS_uu_t21 = 2*ddxk_utau_1k
    # RHS_uu_t22 = -2*tau1k_dudx_k
    
    opt=0
    # if (opt==1):
    #     RHS_uu = RHS_uu_t1 + RHS_uu_t2
    # else:
    #     RHS_uu = RHS_uu_t1 + RHS_uu_t21 + RHS_uu_t22

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

    a1 = np.divide(compute_inner_prod(reig, ueig), Rho)

    
    ############################
    # Balance Equation for v'v'
    ############################

    # Should not have rho_bsfl for Boussinesq!!!!!
    LHS_vv = 2*omega_i*Rs_vv
    
    RHS_vv_t1 = -2.*compute_inner_prod(veig, dpdy)

    # Compute Laplacian of u'u'
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

    # LHS: Should not have rho_bsfl for Boussinesq!!!!!
    LHS_ww = 2*omega_i*Rs_ww

    # RHS
    #RHS_ww_t1 = -2.*compute_inner_prod(weig, dpdz)
    RHS_ww_t11 = -2.*( compute_inner_prod(weig, dpdz) + compute_inner_prod(peig, dwdz) )
    RHS_ww_t12 = 2.*compute_inner_prod(peig, dwdz)
    
    RHS_ww_t2 = 2.*np.multiply(a3, dpdz_bsfl)

    # Compute Laplacian of w'w'
    Lap_ww = 2.*( compute_inner_prod(dwdx, dwdx) + compute_inner_prod(dwdy, dwdy) + compute_inner_prod(dwdz, dwdz) ) + \
             2.*( compute_inner_prod(weig, d2wdx2) + compute_inner_prod(weig, d2wdy2) + compute_inner_prod(weig, d2wdz2) )
    Lap_ww = Lap_ww/Re

    # Compute dissipation of w'w'
    dissip_ww = 2.*( compute_inner_prod(dwdx, dwdx) + compute_inner_prod(dwdy, dwdy) + compute_inner_prod(dwdz, dwdz) )
    dissip_ww = dissip_ww/Re

    # RHS_ww_t3 = 2.*compute_inner_prod(weig, dTau3k_dxk)

    # # Write RHS_ww_t3 as 2*( d/dx_k(w'tau'_3k) - tau'_3k dw'_k/dx_k )
    # ddxk_wtau_3k = 1/Re*( compute_inner_prod(dwdx, dudz) + compute_inner_prod(dwdx, dwdx) + compute_inner_prod(weig, d2udxdz) + compute_inner_prod(weig, d2wdx2) ) +\
    #                1/Re*( compute_inner_prod(dwdy, dwdy) + compute_inner_prod(dwdy, dvdz) + compute_inner_prod(weig, d2wdy2) + compute_inner_prod(weig, d2vdydz) ) +\
    #                2/Re*( compute_inner_prod(dwdz, dwdz) + compute_inner_prod(weig, d2wdz2) )

    # tau3k_dwdx_k = 1/Re*( compute_inner_prod(dudz, dwdx) + compute_inner_prod(dwdx, dwdx) ) +\
    #                1/Re*( compute_inner_prod(dwdy, dwdy) + compute_inner_prod(dvdz, dwdy) ) +\
    #                2/Re*( compute_inner_prod(dwdz, dwdz) )

    # RHS_ww_t31 = 2*ddxk_wtau_3k
    # RHS_ww_t32 = -2*tau3k_dwdx_k

    #if (opt==1):
    #    RHS_ww = RHS_ww_t1 + RHS_ww_t2 + RHS_ww_t3
    #else:
    #    RHS_ww = RHS_ww_t1 + RHS_ww_t2 + RHS_ww_t31 + RHS_ww_t32

    #RHS_ww = RHS_ww_t1 + RHS_ww_t2 + Lap_ww - dissip_ww
    RHS_ww = RHS_ww_t11 + RHS_ww_t12 + RHS_ww_t2 + Lap_ww - dissip_ww

    print("")
    print("Check energy balancefor w'w' Reynolds stress equation")
    print("-----------------------------------------------------")
    energy_bal_ww = np.amax(np.abs(LHS_ww-RHS_ww))
    print("Energy balance Reynolds stress w'w' = ", energy_bal_ww)
    print("")

    ############################
    # Balance Equation for u'v'
    ############################

    # LHS
    LHS_uv = 2*omega_i*Rs_uv

    # RHS
    RHS_uv_t1 = -compute_inner_prod(ueig, dpdy)
    RHS_uv_t2 = -compute_inner_prod(veig, dpdx)

    # Compute Laplacian of u'v'
    Lap_uv = 2.0*( compute_inner_prod(dudx, dvdx) + compute_inner_prod(dudy, dvdy) + compute_inner_prod(dudz, dvdz) ) + \
             compute_inner_prod(ueig, d2vdx2) + compute_inner_prod(ueig, d2vdy2) + compute_inner_prod(ueig, d2vdz2) + \
             compute_inner_prod(veig, d2udx2) + compute_inner_prod(veig, d2udy2) + compute_inner_prod(veig, d2udz2) 
    Lap_uv = Lap_uv/Re

    # Compute "dissipation" of u'v' ===> Note that for alpha = beta, ueig = veig such that this is a dissipation
    dissip_uv = 2.*( compute_inner_prod(dudx, dvdx) + compute_inner_prod(dudy, dvdy) + compute_inner_prod(dudz, dvdz) )
    dissip_uv = dissip_uv/Re

    RHS_uv = RHS_uv_t1 + RHS_uv_t2 + Lap_uv - dissip_uv
        
    print("")
    print("Check energy balancefor u'v' Reynolds stress equation")
    print("-----------------------------------------------------")
    energy_bal_uv = np.amax(np.abs(LHS_uv-RHS_uv))
    print("Energy balance Reynolds stress u'v' = ", energy_bal_uv)
    print("")


    ######################################
    # MODEL TERMS
    ######################################

    #
    # 1) u'u' equation
    #
    
    Cr1 = 1.
    pres_strain_mod_uu = 2./3.*Cr1*np.multiply(a3, dpdz_bsfl)
    # this term = 0 for RT: mod_try = 2./3.*Cr1*np.multiply(a1, dpdz_bsfl)
    #pres_strain_mod22 = np.divide(pres_strain_mod, Rho)

    #
    # 2) w'w' equation
    #
    
    pres_strain_mod_ww = -4./3.*Cr1*np.multiply(a3, dpdz_bsfl)
    pres_strain_mod_ww_girimaji = (-2. + 4./3.*0.54)*a3

    B = 1.
    dazdz = np.matmul(D1, a3)
    #dazdz22 = np.divide( ( compute_inner_prod(reig, dwdz) + compute_inner_prod(weig, Dr) ), Rho ) - np.divide( np.multiply(Rhop, compute_inner_prod(reig, weig)), rho2)
    #print("np.max(np.abs(dazdz-dazdz22)) = ", np.max(np.abs(dazdz-dazdz22)))
    #input("checking dadz")
    
    l_taylor_nondim = iarr.l_taylor_dim/Lref
    l_taylor_nondim2 = np.multiply(l_taylor_nondim, l_taylor_nondim)

    l_integr_nondim = iarr.l_integr_dim/Lref
    l_integr_nondim2 = np.multiply(l_integr_nondim, l_integr_nondim)

    if ( np.amax(np.abs(iarr.l_taylor_dim)) != 0.0 and np.amax(np.abs(iarr.l_integr_dim)) != 0.0 ): 
        pass
    else:
        sys.exit("Something not right with the length scales!!!!!")    

    #print("l_taylor_nondim = ", l_taylor_nondim)

    pres_diffusion_mod_ww = 2.*np.multiply( Rho, np.multiply(l_taylor_nondim2, dazdz) )
    pres_diffusion_mod_ww = np.matmul(D1, pres_diffusion_mod_ww)
    
    #pres_diffusion_mod_ww22 = 2.*np.multiply( Rho, np.multiply(l_integr_nondim2, dazdz) )
    #pres_diffusion_mod_ww22 = np.matmul(D1, pres_diffusion_mod_ww22)
    
    #pres_diffusion_mod_ww22 = 2.*np.multiply( 1., np.multiply(l_taylor_nondim2, dazdz) )
    #pres_diffusion_mod_ww22 = np.matmul(D1, pres_diffusion_mod_ww22)

    ######################################
    # PLOT REYNOLDS STRESSES BALANCE EQ.
    ######################################

    #######
    # u'u'
    #######
    
    ptn = plt.gcf().number + 1

    ymin, ymax = FindResonableYminYmax(Rs_ww, y)
    
    f  = plt.figure(ptn)
    ax = plt.subplot(111)
    ax.plot(np.real(LHS_uu), y, 'k', linewidth=1.5, label=r"LHS $(\overline{u'u'})$")
    ax.plot(np.real(RHS_uu), y, 'g--', linewidth=1.5, label=r"RHS $(\overline{u'u'})$")

    ax.plot(np.real(RHS_uu_t1), y, 'r', linewidth=1.5, label=r"$-2\overline{u'\dfrac{\partial p'}{\partial x}}$ (RHS)")

    ax.plot(np.real(Lap_uu), y, 'b', linewidth=1.5, label=r"$\dfrac{1}{Re} \nabla^2 \overline{u'u'}$ (RHS)")
    ax.plot(np.real(dissip_uu), y, 'm', linewidth=1.5, label=r"$\dfrac{2}{Re} \overline{\nabla u' \cdot \nabla u'}$ (RHS)")    

    # if (opt==1):
    #     ax.plot(np.real(RHS_uu_t2), y, 'b', linewidth=1.5, label=r"$2\overline{u'\dfrac{\partial \tau'_{1k}}{\partial x_k}}$ (RHS)")
    # else:
    #     #ax.plot(np.real(RHS_uu_t2), y, 'b', linewidth=1.5, label=r"$(2\overline{u'\dfrac{\partial \tau'_{1k}}{\partial x_k}})$ (RHS)")
    #     ax.plot(np.real(RHS_uu_t21), y, 'm', linewidth=1.5, label=r"$ 2 \dfrac{\partial}{\partial x_k} \overline{u'\tau'_{1k}}$ (RHS)")
    #     ax.plot(np.real(RHS_uu_t22), y, 'c', linewidth=1.5, label=r"$-2 \overline{\tau'_{1k} \dfrac{\partial u'}{\partial x_k}}$ (RHS)")
    #     ax.plot(np.real(RHS_uu_t21+RHS_uu_t22), y, 'y-.', linewidth=1.5, label=r"viscous (RHS)")
    
    plt.xlabel("Reynolds stresses balance eq. for u'u'", fontsize=18)
    plt.ylabel("z", fontsize=18)
    plt.gcf().subplots_adjust(left=0.17)
    plt.gcf().subplots_adjust(bottom=0.15)
    plt.ylim([ymin, ymax])
    
    plt.legend(loc="upper right")
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15),   # box with lower left corner at (x, y)
              ncol=3, fancybox=True, shadow=True, fontsize=10)
    
    f.show()

    #
    # Model terms vs exact from u'u' equation
    #

    # d/dx(p'u')
    ddx_pu = ia*compute_inner_prod(peig, ueig)
    # p'*du'/dx
    pdudx = compute_inner_prod(peig, dudx)
    # u'*dp'/dx
    udpdx = compute_inner_prod(ueig, dpdx)
    
    ptn = plt.gcf().number + 1

    f  = plt.figure(ptn)
    ax = plt.subplot(111)

    # Exact
    ax.plot(np.real(ddx_pu),y, 'k', linewidth=1.5, label=r"$\dfrac{\partial}{\partial x} \left( p'u' \right )$")
    ax.plot(np.real(pdudx), y, 'g--', linewidth=1.5, label=r"$p'\dfrac{\partial u'}{\partial x}$")
    ax.plot(np.real(udpdx), y, 'r', linewidth=1.5, label=r"$u'\dfrac{\partial p'}{\partial x}$")

    # Model
    ax.plot(np.real(pres_strain_mod_uu), y, 'm-.', linewidth=1.5, label=r"ps model ($\dfrac{2}{3}C_{r1}a_z \dfrac{\partial \bar{p}}{\partial z}$)")
    
    plt.xlabel("Rxx balance eq. (u'u'): exact vs. model", fontsize=18)
    plt.ylabel("z", fontsize=18)
    plt.gcf().subplots_adjust(left=0.17)
    plt.gcf().subplots_adjust(bottom=0.15)
    plt.ylim([ymin, ymax])
    
    plt.legend(loc="upper right")
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15),   # box with lower left corner at (x, y)
              ncol=4, fancybox=True, shadow=True, fontsize=10)
    
    f.show()
    
    ############
    # Plot components of tau'1k du'/dx_k
    ############

    ptn = plt.gcf().number + 1

    ymin, ymax = FindResonableYminYmax(Rs_ww, y)
    
    f  = plt.figure(ptn)
    ax = plt.subplot(111)
    ax.plot(np.real(2/Re*( compute_inner_prod(dudx, dudx) )), y, 'k', linewidth=1.5, label=r"$\overline{u'u'}$: $\dfrac{2}{Re}\overline{\dfrac{\partial u'}{\partial x}\dfrac{\partial u'}{\partial x}}$")
    ax.plot(np.real(1/Re*( compute_inner_prod(dudy, dudy) )), y, 'r', linewidth=1.5, label=r"$\overline{u'u'}$: $\dfrac{1}{Re}\overline{\dfrac{\partial u'}{\partial y}\dfrac{\partial u'}{\partial y}}$")
    ax.plot(np.real(1/Re*( compute_inner_prod(dudz, dudz) )), y, 'b', linewidth=1.5, label=r"$\overline{u'u'}$: $\dfrac{1}{Re}\overline{\dfrac{\partial u'}{\partial z}\dfrac{\partial u'}{\partial z}}$")
    # cross-terms
    ax.plot(np.real(1/Re*( compute_inner_prod(dudy, dvdx) )), y, 'c-.', linewidth=1.5, label=r"$\overline{u'u'}$: $\dfrac{1}{Re}\overline{\dfrac{\partial u'}{\partial y}\dfrac{\partial v'}{\partial x}}$")
    ax.plot(np.real(1/Re*( compute_inner_prod(dudz, dwdx) )), y, 'm', linewidth=1.5, label=r"$\overline{u'u'}$: $\dfrac{1}{Re}\overline{\dfrac{\partial u'}{\partial z}\dfrac{\partial w'}{\partial x}}$")
    
        
    plt.xlabel("Reynolds stresses balance eq. for u'u' ==> some components", fontsize=18)
    plt.ylabel("z", fontsize=18)
    plt.gcf().subplots_adjust(left=0.17)
    plt.gcf().subplots_adjust(bottom=0.15)
    plt.ylim([ymin, ymax])
    
    plt.legend(loc="upper right")
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15),   # box with lower left corner at (x, y)
              ncol=3, fancybox=True, shadow=True, fontsize=10)
    
    f.show()


    
    #######
    # w'w'
    #######
    
    ptn = plt.gcf().number + 1

    ymin, ymax = FindResonableYminYmax(Rs_ww, y)
    
    f  = plt.figure(ptn)
    ax = plt.subplot(111)
    ax.plot(np.real(LHS_ww), y, 'k', linewidth=1.5, label=r"LHS $(\overline{w'w'})$")
    ax.plot(np.real(RHS_ww), y, 'g--', linewidth=1.5, label=r"RHS $(\overline{w'w'})$")

    max_pd = np.max(np.abs(RHS_ww_t11))
    max_pd_mod = np.max(np.abs(pres_diffusion_mod_ww))

    scale_pd = max_pd/max_pd_mod
    print("scale_pd = ", scale_pd)
    
    #ax.plot(np.real(RHS_ww_t1), y, 'r', linewidth=1.5, label=r"$-2\overline{w'\dfrac{\partial p'}{\partial z}}$ (RHS)")
    ax.plot(np.real(RHS_ww_t11), y, 'b', linewidth=1.5, label=r"$-2 \dfrac{\partial}{\partial z} \overline{p'w'}$ (RHS)") # --> pressure diffusion
    ax.plot(np.real(RHS_ww_t12), y, 'c', linewidth=1.5, label=r"$2\overline{p'\dfrac{\partial w'}{\partial z}}$ (RHS)")   # --> pressure strain

    # mode terms
    label_pd_mod = "pd model ( scaling = %s )" % str('{:.2e}'.format(scale_pd))
    
    ax.plot(np.real(pres_diffusion_mod_ww)*scale_pd, y, 'b-.', linewidth=1.5, label=label_pd_mod)
    ax.plot(np.real(pres_strain_mod_ww), y, 'c-.', linewidth=1.5, label=r"ps model")
    ax.plot(np.real(pres_strain_mod_ww_girimaji), y, 'c--', linewidth=1.5, label=r"ps model (Girimaji)")

    ax.plot(np.real(RHS_ww_t2), y, 'm-.', linewidth=1.5, label=r"$2a_z \dfrac{\partial \bar{p}}{\partial z}$ (RHS)")

    ax.plot(np.real(Lap_ww), y, 'm', linewidth=1.5, label=r"$\dfrac{1}{Re} \nabla^2 \overline{w'w'}$ (RHS)")
    ax.plot(np.real(dissip_ww), y, 'y', linewidth=1.5, label=r"$\dfrac{2}{Re} \overline{\nabla w' \cdot \nabla w'}$ (RHS)")

    #ax.plot(np.real(dazdz), y, 'y', linewidth=1.5, color='orange', label=r"dazdz")

    #try_var = np.multiply(l_taylor_nondim2, np.matmul(D1, dazdz))
    #try_var = np.multiply(l_integr_nondim2, np.matmul(D1, dazdz))
    #ax.plot(np.real(try_var), y, 'y', linewidth=3, color='orange', label=r"dazdz")

    #pres_diffusion_mod_ww, pres_strain_mod_ww

    #if (opt==1):
    #    ax.plot(np.real(RHS_ww_t3), y, 'm', linewidth=1.5, label=r"$2\overline{w'\dfrac{\partial \tau'_{3k}}{\partial x_k}}$ (RHS)")
    #else:
        #ax.plot(np.real(RHS_ww_t3), y, 'm', linewidth=1.5, label=r"$(2\overline{w'\dfrac{\partial \tau'_{3k}}{\partial x_k}})$ (RHS)")
    #    ax.plot(np.real(RHS_ww_t31), y, 'c', linewidth=1.5, label=r"$2 \dfrac{\partial}{\partial x_k} \overline{w'\tau'_{3k}}$ (RHS)")
    #    ax.plot(np.real(RHS_ww_t32), y, 'y', linewidth=1.5, label=r"$-2 \overline{\tau'_{3k} \dfrac{\partial w'}{\partial x_k}}$ (RHS)")
    #    ax.plot(np.real(RHS_ww_t31+RHS_ww_t32), y, 'k-.', linewidth=1.5, label=r"viscous (RHS)")
    
    #ax.plot(np.real(pres_diffusion_mod_ww22), y, 'm--', linewidth=1.5, label=r"pd model (integr length)")
    #ax.plot(np.real(pres_strain_mod_ww+pres_diffusion_mod_ww), y, 'm', linewidth=1.5, label=r"ps model + pd model")
    
    plt.xlabel("Reynolds stresses balance eq. for w'w'", fontsize=18)
    plt.ylabel("z", fontsize=18)
    plt.gcf().subplots_adjust(left=0.17)
    plt.gcf().subplots_adjust(bottom=0.15)
    plt.ylim([ymin, ymax])
    #plt.ylim([-15, 30])
    #plt.ylim([-5, 5])
    
    plt.legend(loc="upper right")
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15),   # box with lower left corner at (x, y)
              ncol=3, fancybox=True, shadow=True, fontsize=8)
    
    f.show()

    #
    # Model terms vs exact from w'w' equation
    #

    # d/dz(p'w')
    ddz_pw = np.matmul(D1, compute_inner_prod(peig, weig))
    # p'*dw'/dz
    pdwdz = compute_inner_prod(peig, dwdz)
    # w'*dp'/dz
    wdpdz = compute_inner_prod(weig, dpdz)
    
    ptn = plt.gcf().number + 1

    f  = plt.figure(ptn)
    ax = plt.subplot(111)

    # Exact terms
    ax.plot(np.real(ddz_pw),y, 'k', linewidth=1.5, label=r"$\dfrac{\partial}{\partial z} \left( p'w' \right )$")
    ax.plot(np.real(pdwdz), y, 'g--', linewidth=1.5, label=r"$p'\dfrac{\partial w'}{\partial z}$")
    ax.plot(np.real(wdpdz), y, 'r', linewidth=1.5, label=r"$w'\dfrac{\partial p'}{\partial z}$")

    # Model terms
    ax.plot(np.real(pres_strain_mod_ww), y, 'c-.', linewidth=1.5, label=r"ps model")

    #fac_mod = 1. #max_1/np.amax(np.abs(pres_diffusion_mod_ww))
    #pd_mod  = "pressure diffusion model, sign? ( scaling = %s )" % str('{:.2e}'.format(fac_mod))
    pd_mod  = "pressure diffusion model"
    ax.plot(np.real(pres_diffusion_mod_ww), y, 'm-.', linewidth=1.5, label=pd_mod)
    #ax.plot(np.real(pres_diffusion_mod_ww)*fac_mod, y, 'm-.', linewidth=1.5, label=pd_mod)

    
    plt.xlabel("Rzz balance eq. (w'w'): exact vs. model", fontsize=18)
    plt.ylabel("z", fontsize=18)
    plt.gcf().subplots_adjust(left=0.17)
    plt.gcf().subplots_adjust(bottom=0.15)
    plt.ylim([ymin, ymax])
    
    plt.legend(loc="upper right")
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15),   # box with lower left corner at (x, y)
              ncol=3, fancybox=True, shadow=True, fontsize=10)
    
    f.show()



    #######
    # u'v'
    #######

    #RHS_uv_t1 = -compute_inner_prod(ueig, dpdy)
    #RHS_uv_t2 = -compute_inner_prod(veig, dpdx)
    
    ptn = plt.gcf().number + 1

    ymin, ymax = FindResonableYminYmax(Rs_uv, y)
    
    f  = plt.figure(ptn)
    ax = plt.subplot(111)
    ax.plot(np.real(LHS_uv), y, 'k', linewidth=1.5, label=r"LHS $(\overline{u'v'})$")
    ax.plot(np.real(RHS_uv), y, 'g--', linewidth=1.5, dashes=[5, 5], label=r"RHS $(\overline{u'v'})$")

    ax.plot(np.real(RHS_uv_t1), y, 'r', linewidth=1.5, label=r"$-\overline{u'\dfrac{\partial p'}{\partial y}}$ (RHS)")
    ax.plot(np.real(RHS_uv_t2), y, 'b--', linewidth=1.5, dashes=[5, 5], label=r"$-\overline{v'\dfrac{\partial p'}{\partial x}}$ (RHS)")

    ax.plot(np.real(Lap_uv), y, 'm', linewidth=1.5, label=r"$\dfrac{1}{Re} \nabla^2 \overline{u'v'}$ (RHS)")
    ax.plot(np.real(dissip_uv), y, 'c', linewidth=1.5, label=r"$\dfrac{2}{Re} \overline{\nabla u' \cdot \nabla v'}$ (RHS)")    
    
    plt.xlabel("Reynolds stresses balance eq. for u'v'", fontsize=18)
    plt.ylabel("z", fontsize=18)
    plt.gcf().subplots_adjust(left=0.17)
    plt.gcf().subplots_adjust(bottom=0.15)
    plt.ylim([ymin, ymax])
    
    plt.legend(loc="upper right")
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15),   # box with lower left corner at (x, y)
              ncol=3, fancybox=True, shadow=True, fontsize=10)
    
    f.show()

    #input("Check Reynolds stress energy balance ---> u'u' and w'w'")


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

    # Get ymin-ymax
    ymin, ymax = FindResonableYminYmax(iarr.l_taylor_dim, y)

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
    ymin, ymax = FindResonableYminYmax(iarr.l_integr_dim, y)

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

    input("length scales")


def compute_disturbance_kin_energy(ueig, veig, weig):

    # Disturbance kinetic energy
    uu = compute_inner_prod(ueig, ueig)
    vv = compute_inner_prod(veig, veig)
    ww = compute_inner_prod(weig, weig)
    ke_dist = 0.5*( uu + vv + ww )

    return ke_dist


def compute_velocity_pressure_correlations(ueig, veig, weig, peig, alpha, beta, D1):

    # Velocity pressure correlation
    vpc1 = -compute_inner_prod(ueig, 1j*alpha*peig)
    vpc2 = -compute_inner_prod(veig, 1j*beta*peig)
    vpc3 = -compute_inner_prod(weig, np.matmul(D1, peig))

    vpc_tot = vpc1 + vpc2 + vpc3

    return vpc1, vpc2, vpc3, vpc_tot

def compute_disturbance_viscous_transport(ueig, veig, weig, peig, reig, Re, alpha, beta, D1):

    dudx,dudy,dudz,dvdx,dvdy,dvdz,dwdx,dwdy,dwdz,dpdx,dpdy,dpdz,drdx,drdy,drdz = get_eigenfunction_derivatives(ueig, veig, weig, peig, reig, alpha, beta, D1)
    
    utau11 = 2.*compute_inner_prod(ueig, dudx)/Re
    vtau21 = ( compute_inner_prod(veig, dudy) + compute_inner_prod(veig, dvdx) )/Re
    wtau31 = ( compute_inner_prod(weig, dudz) + compute_inner_prod(weig, dwdx) )/Re
    
    utau12 = ( compute_inner_prod(ueig, dudy) + compute_inner_prod(ueig, dvdx) )/Re
    vtau22 = 2.*compute_inner_prod(veig, dvdy)/Re
    wtau32 = ( compute_inner_prod(weig, dwdy) + compute_inner_prod(weig, dvdz) )/Re
    
    utau13 = ( compute_inner_prod(ueig, dudz) + compute_inner_prod(ueig, dwdx) )/Re
    vtau23 = ( compute_inner_prod(veig, dvdz) + compute_inner_prod(veig, dwdy) )/Re
    wtau33 = 2.*compute_inner_prod(weig, dwdz)/Re

    visc_trans_dist = 1j*alpha*( utau11 + vtau21 + wtau31 ) + 1j*beta*( utau12 + vtau22 + wtau32) + np.matmul(D1, utau13+vtau23+wtau33 )

    return visc_trans_dist


def get_eigenfunction_derivatives(ueig, veig, weig, peig, reig, alpha, beta, D1):

    dudx = 1j*alpha*ueig
    dudy = 1j*beta*ueig
    dudz = np.matmul(D1, ueig)
    
    dvdx = 1j*alpha*veig
    dvdy = 1j*beta*veig
    dvdz = np.matmul(D1, veig)
    
    dwdx = 1j*alpha*weig
    dwdy = 1j*beta*weig
    dwdz = np.matmul(D1, weig)

    dpdx = 1j*alpha*peig
    dpdy = 1j*beta*peig
    dpdz = np.matmul(D1, peig)

    drdx = 1j*alpha*reig
    drdy = 1j*beta*reig
    drdz = np.matmul(D1, reig)    

    return dudx,dudy,dudz,dvdx,dvdy,dvdz,dwdx,dwdy,dwdz,dpdx,dpdy,dpdz,drdx,drdy,drdz

# dudx,dudy,dudz,dvdx,dvdy,dvdz,dwdx,dwdy,dwdz,dpdx,dpdy,dpdz,drdx,drdy,drdz = get_eigenfunction_derivatives(ueig, veig, weig, peig, reig, alpha, beta, D1)

def get_eigenfunction_2nd_derivatives(ueig, veig, weig, peig, reig, alpha, beta, D2):

    ia2 = ( 1j*alpha )**2.
    ib2 = ( 1j*beta )**2.

    iab = ( 1j*alpha )*( 1j*beta )

    d2udx2 = ia2*ueig
    d2udy2 = ib2*ueig
    d2udz2 = np.matmul(D2, ueig)

    d2vdx2 = ia2*veig
    d2vdy2 = ib2*veig
    d2vdz2 = np.matmul(D2, veig)

    d2wdx2 = ia2*weig
    d2wdy2 = ib2*weig
    d2wdz2 = np.matmul(D2, weig)

    d2pdx2 = ia2*peig
    d2pdy2 = ib2*peig
    d2pdz2 = np.matmul(D2, peig)

    d2rdx2 = ia2*reig
    d2rdy2 = ib2*reig
    d2rdz2 = np.matmul(D2, reig)

    return d2udx2, d2udy2, d2udz2, d2vdx2, d2vdy2, d2vdz2, d2wdx2, d2wdy2, d2wdz2, d2pdx2, d2pdy2, d2pdz2, d2rdx2, d2rdy2, d2rdz2
    

def plot_disturbance_dissipation(y, epsilon_dist, epsilon_incomp_dist, epsilon_comp_dist, epsilon_tilde_dist, epsilon_2tilde_dist, epsilon_b, epsilon_b_tilde):

    #######################################
    # PLOTS
    #######################################
    
    ptn = plt.gcf().number + 1
    
    # Plot various components of dissipation
    # --------------------------------------

    ymin, ymax = FindResonableYminYmax(epsilon_dist, y)
    
    f = plt.figure(ptn)
    ax = plt.subplot(111)
    ax.plot(np.real(epsilon_dist),        y, 'k', linewidth=1.5, label=r"$\epsilon = \epsilon_{incomp} + \epsilon_{comp}$")
    ax.plot(np.real(epsilon_incomp_dist), y, 'g--', linewidth=1.5, dashes=[5, 5], label=r"$\epsilon_{incomp} = \tilde{\epsilon} + \tilde{\tilde{\epsilon}}$")
    ax.plot(np.real(epsilon_comp_dist),   y, 'm', linewidth=1.5, label=r"$\epsilon_{comp}$")
    #
    ax.plot(np.real(epsilon_tilde_dist),  y, 'r', linewidth=1.5, label=r"$\tilde{\epsilon}$")    
    ax.plot(np.real(epsilon_2tilde_dist), y, 'b--', linewidth=1.5, dashes=[5, 5], label=r"$\tilde{\tilde{\epsilon}}$")

    # "Vorticity" dissipation
    ax.plot(np.real(epsilon_tilde_dist-epsilon_2tilde_dist), y, 'c', linewidth=1.5, label=r"$\tilde{\epsilon} - \tilde{\tilde{\epsilon}}$")

    # Dissipation of b-equation
    ax.plot(np.real(epsilon_b), y, 'y--', linewidth=1.5, label=r"$\epsilon_b$")
    ax.plot(np.real(epsilon_b_tilde), y, 'y', linewidth=1.5, label=r"$\tilde{\epsilon}_b$")

    plt.xlabel("dissipation components", fontsize=14)
    plt.ylabel('z', fontsize=16)
    plt.gcf().subplots_adjust(left=0.18)
    plt.gcf().subplots_adjust(bottom=0.15)
    #plt.xlim([xmin, xmax])
    plt.ylim([ymin, ymax])

    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15),   # box with lower left corner at (x, y)
              ncol=3, fancybox=True, shadow=True, fontsize=10)
    
    f.show()

    input("Components of disturbance dissipation")


def calc_min_dy(y, boussi, bsfl_ref):

    ny = len(y)
    
    if (boussi==-2): # dimensional solver
        y = y
    else:            # nondimensional solvers
        y = y*bsfl_ref.Lref

    dymin = 1e10   
    for i in range(0, ny-1):
        dy = y[i+1]-y[i]
        if ( dy < dymin ):
            dymin = dy

    print("")
    print("dy_min (dimensional) = ", dymin)
    print("")

    # Compute model dissipation for b-equation
    #Cb1 = 2.0
    #t1_ = np.divide(epsilon_dist, KE_dist)
    #t2_ = np.divide(b, Rho)
    #epsilon_b_model = Cb1/2.0*np.multiply(t1_, t2_)
    #print("Dissip_disturb = ", Dissip_disturb)

    
def compute_a_eq_ener_bal_boussi_Sc_1_NEW(ueig, veig, weig, peig, reig, D1, D2, y, alpha, beta, omega_i, bsfl, bsfl_ref, mob, rt_flag, KE_dist, epsilon_dist, epsil_tild_min_2tild_dist, iarr):

    #plt.close('all')

    ny = len(ueig)
    
    # Get some quantities to compute the balances
    Mu, Mup, Rho, Rhop, Rhopp, rho2, rho3, rho_inv, rho2_inv, rho3_inv = get_baseflow_and_derivatives(bsfl, mob)
    ia, ib, ia2, ib2, iab, Du, Dv, Dw, Dr, D2u, D2v, D2w, D2r = get_eigenfunctions_quantities(alpha, beta, D1, D2, ueig, veig, weig, reig)
    
    Sc = bsfl_ref.Sc
    Re = bsfl_ref.Re
    Fr = bsfl_ref.Fr

    Lref = bsfl_ref.Lref
    Uref = bsfl_ref.Uref
    muref = bsfl_ref.muref
    rhoref = bsfl_ref.rhoref

    div = ( 1j*alpha*ueig + 1j*beta*veig + Dw )

    v = -np.divide(reig, rho2)

    dpdx = (1j*alpha)*peig
    dpdy = (1j*beta)*peig
    dpdz = np.matmul(D1, peig) 
    Dp  = dpdz

    # Specific volume derivatives
    dvdx = (1j*alpha)*v
    dvdy = (1j*beta)*v
    dvdz = np.matmul(D1, v)

    
    dwdx    = (1j*alpha)*weig
    dwdy    = (1j*beta)*weig
    dwdz    = Dw

    drdx    = (1j*alpha)*reig
    drdy    = (1j*beta)*reig
    drdz    = Dr

    d2wdx2  = (1j*alpha)**2.*weig
    d2wdy2  = (1j*beta)**2.*weig
    d2wdz2  = D2w

    d2rdx2  = (1j*alpha)**2.*reig
    d2rdy2  = (1j*beta)**2.*reig
    d2rdz2  = D2r

    # LHS for a-equation
    LHS_aEq = 2*omega_i*compute_inner_prod(reig, weig)

    # Compute Laplacian of rho'w'
    Lap_a = 2.0*( compute_inner_prod(drdx, dwdx) + compute_inner_prod(drdy, dwdy) + compute_inner_prod(drdz, dwdz) ) + \
             compute_inner_prod(reig, d2wdx2) + compute_inner_prod(reig, d2wdy2) + compute_inner_prod(reig, d2wdz2) + \
             compute_inner_prod(weig, d2rdx2) + compute_inner_prod(weig, d2rdy2) + compute_inner_prod(weig, d2rdz2) 
    Lap_a = Lap_a/Re

    # Compute "dissipation" of a-equation
    dissip_a = 2.*( compute_inner_prod(drdx, dwdx) + compute_inner_prod(drdy, dwdy) + compute_inner_prod(drdz, dwdz) )
    dissip_a = dissip_a/Re

    # RHS: inviscid terms
    #RHS_a_t1 = -compute_inner_prod(reig, dpdz)
    RHS_a_t11 = -compute_inner_prod(reig, dpdz)-compute_inner_prod(peig, drdz)
    RHS_a_t12 = compute_inner_prod(peig, drdz)
    
    RHS_a_t2 = -np.multiply(Rhop, compute_inner_prod(weig, weig))
    RHS_a_t3 = -compute_inner_prod(reig, reig)/Fr**2.

    #RHS_aEq  = RHS_a_t1 + RHS_a_t2 + RHS_a_t3 + Lap_a - dissip_a
    RHS_aEq  = RHS_a_t11 + RHS_a_t12 + RHS_a_t2 + RHS_a_t3 + Lap_a - dissip_a

    energy_bal_aEq  = np.amax(np.abs(LHS_aEq-RHS_aEq))
    print("Energy balance (RT) a-equation (BOUSSINESQ ----> ONLY FOR Sc=1 --- NEW):", energy_bal_aEq)

    ####################
    # Model Terms
    ####################
    dpdz_bsfl = -Rho/Fr**2*1
    
    az = np.divide(compute_inner_prod(reig, weig), Rho)
    b  = np.divide(compute_inner_prod(reig, reig), rho2 )
    dbdz = np.matmul(D1, b)

    l_taylor_nondim = iarr.l_taylor_dim/Lref
    l_taylor_nondim2 = np.multiply(l_taylor_nondim, l_taylor_nondim)

    pres_dissipation_model = 1./3.*np.multiply(b, dpdz_bsfl)
    pres_diffusion_model = -np.matmul(D1, np.multiply(l_taylor_nondim2, dbdz))

    ####################
    # PLOTS
    ####################

    # GET ymin-ymax
    ymin, ymax = FindResonableYminYmax(RHS_aEq, y)

    ptn = plt.gcf().number + 1
    
    f = plt.figure(ptn)
    ax = plt.subplot(111)

    #plt.plot(np.real(RHS_a_t1), y, 'b--', linewidth=1.5, label=r"$-\overline{ \rho' \dfrac{\partial p'}{\partial z} }$")
    
    plt.plot(np.real(RHS_a_t11), y, 'b', linewidth=1.5, label=r"$-\dfrac{\partial}{\partial z} \left( \overline{ \rho'p' } \right)$") ## pressure diffusion
    plt.plot(np.real(RHS_a_t12), y, 'c', linewidth=1.5, label=r"$\overline{ p'\dfrac{\partial \rho'}{\partial z} }$")                 ## pressure dissipation
    
    plt.plot(np.real(RHS_a_t2), y, 'g', linewidth=1.5, label=r"$-\dfrac{\partial \bar{\rho}}{\partial z}\overline{w'w'}$")
    plt.plot(np.real(RHS_a_t3), y, 'y', linewidth=1.5, label=r"$-\overline{\rho'\rho'}g/Fr^2$")
    plt.plot(np.real(Lap_a),    y, 'm-.', linewidth=1.5, label=r"$\dfrac{1}{Re}\nabla^2 \overline{\rho'w'}$")
    plt.plot(np.real(dissip_a), y, 'm', linewidth=1.5, label=r"$2\dfrac{\partial \rho'}{\partial x_j}\dfrac{\partial w'}{\partial x_j}$")    

    plt.plot(np.real(RHS_aEq), y, 'r', linewidth=1.5, label="RHS")
    plt.plot(np.real(LHS_aEq), y, 'k--', linewidth=1.5, label=r"LHS: $\dfrac{\partial}{\partial t} \overline{\rho'w'}$")

    max_pd = np.max(np.abs(RHS_a_t11))
    max_pd_model = np.max(np.abs(pres_diffusion_model))
    scale_pd = max_pd/max_pd_model
    
    label_pd_mod = "pd model ( scaling = %s )" % str('{:.2e}'.format(scale_pd))
    plt.plot(np.real(pres_diffusion_model)*scale_pd, y, 'b-.', linewidth=1.5, label=label_pd_mod)
    plt.plot(np.real(pres_dissipation_model), y, 'c-.', linewidth=1.5, label="pd model")
    #plt.plot(np.real(pres_dissipation_model+pres_diffusion_model), y, 'c:', linewidth=1.5, label="pressure dissipation + pressure diffusion model")
    
    plt.xlabel(r"a-equation (Boussinesq, Sc=1)", fontsize=14)
    plt.ylabel('y', fontsize=16)
    plt.gcf().subplots_adjust(left=0.16)
    plt.gcf().subplots_adjust(bottom=0.15)
    plt.ylim([ymin, ymax])
    #plt.ylim([-8, 20])
    #plt.ylim([-5, 5])

    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=3, fancybox=True, shadow=True, fontsize=8)

    f.show()

    input("Boussinesq a-equation balance...")


    
# using Test
# using Plots
# using LinearAlgebra: norm

# function dmin_series(lim,x,y,nsamples)
#     dmin = +Inf
#     for _ in 1:nsamples
#         isample = rand(1:length(x))
#         d = norm(lim .- (x[isample],y[isample]))
#         if d < dmin
#             dmin = d
#         end
#     end
#     return dmin
# end

# function find_best_legend_position(plt;nsamples=50)
#     ylims = Plots.ylims(plt)
#     xlims = Plots.xlims(plt)
#     dmin_max = 0.
#     ibest = 0
#     i = 0
#     for lim in Iterators.product(xlims,ylims)
#         i += 1
#         for series in plt.series_list
#             x = series[:x]
#             y = series[:y]
#             dmin = dmin_series(lim,x,y,nsamples)
#             if dmin > dmin_max
#                 dmin_max = dmin
#                 ibest = i
#             end
#         end
#     end
#     ibest == 1 && return :bottomleft
#     ibest == 2 && return :bottomright
#     ibest == 3 && return :topleft
#     return :topright
# end

# function test()

#     x = 0:0.01:2;
#     plt = plot(x,x,label="linear")
#     plt = plot!(x,x.^2,label="quadratic")
#     plt = plot!(x,x.^3,label="cubic")
#     @test find_best_legend_position(plt) == :topleft

#     x = 0:0.01:2;
#     plt = plot(x,-x,label="linear")
#     plt = plot!(x,-x.^2,label="quadratic")
#     plt = plot!(x,-x.^3,label="cubic")
#     @test find_best_legend_position(plt) == :bottomleft

#     x = [0,1,0,1]
#     y = [0,0,1,1]
#     plt = scatter(x,y,xlims=[0.0,1.3],ylims=[0.0,1.3],label="test")
#     @test find_best_legend_position(plt) == :topright
    
#     plt = scatter(x,y,xlims=[-0.3,1.0],ylims=[-0.3,1.0],label="test")
#     @test find_best_legend_position(plt) == :bottomleft

#     plt = scatter(x,y,xlims=[0.0,1.3],ylims=[-0.3,1.0],label="test")
#     @test find_best_legend_position(plt) == :bottomright

#     plt = scatter(x,y,xlims=[-0.3,1.0],ylims=[0.0,1.3],label="test")
#     @test find_best_legend_position(plt) == :topleft

#     true
# end


#input("Press any key to terminate the program")
#sys.exit("debug!!!!!!")


# print("dupdx[0] = ", dupdx[0])
# print("dupdx[-1] = ", dupdx[-1])

# print("dvpdy[0] = ", dvpdy[0])
# print("dvpdy[-1] = ", dvpdy[-1])

# conti_eq = dupdx + dvpdy
# print("conti_eq = ",conti_eq)


# array_check = np.zeros(5)
# array_check[0]=1
# array_check[1]=2
# array_check[2]=3
# array_check[3]=4
# array_check[4]=5
# print("array_check = ",array_check)
# print("array_check[1:5] = ",array_check[1:5])
# input("check array_check")


#plt.plot([1, 2, 3, 4])
#plt.ylabel('some numbers')
#plt.show()


#print("cheb.xc = ",cheb.xc)
#print("cheb.D1 = ",cheb.D1)
#print("cheb.D2 = ",cheb.D2)

#input()


# print("yi = ", yi)
# print("map.y = ", map.y)
# print("map.y[0] = ", map.y[0])
# print("map.y[-1] = ", map.y[-1])

# input('check order of y')

#print("")
#print("map.D1 = ", map.D1)
#print("map.D2 = ", map.D2)

#input()


# g = plt.figure(2)
# plt.plot(k, cp_i)
# plt.plot(k, cm_i)
# plt.xlabel('k')
# plt.ylabel('cp/cm (imag)')
# g.show()

# The input() function is used here to keep the figures alive
#input()

#print("eigvals=",eigvals)
#print("np.shape(eigvects) = ", np.shape(eigvects))


    # for i in range(0, 4):
    #     # Plot eigenfunctions
    #     f = plt.figure(i+10)
    #     #plt.plot(np.abs(eigvects[i*ny:(i+1)*ny, idx_tar1]), y, 'b', label="Amplitude")
    #     #plt.plot(eigvects[i*ny:(i+1)*ny, idx_tar1].real, y, 'r', label="Real part")
    #     #plt.plot(eigvects[i*ny:(i+1)*ny, idx_tar1].imag, y, 'k', label="Imaginary part")
    #     #plt.xlabel(str_vars[i])
    #     #plt.ylabel('y')
    #     plt.plot(y, eigvects[i*ny:(i+1)*ny, idx_tar1].real, 'r', label="Real part")
    #     plt.plot(y, eigvects[i*ny:(i+1)*ny, idx_tar1].imag, 'k', label="Imaginary part")
    #     plt.plot(y, np.abs(eigvects[i*ny:(i+1)*ny, idx_tar1]), 'b', label="Amplitude")

    #     plt.xlabel('y')
    #     plt.ylabel(str_vars[i])
    #     plt.title('Eigenfunction: %s' % (str_vars[i])) #plt.title('f model: T= %d' % (t))
    #     plt.legend(loc="upper left")
    #     f.show()

    # Plot Phi to comprae with Michalke (1964)


    # Phi_vec     = np.zeros(ny, dpc)
    # Phi_vec_new = np.zeros(ny, dpc)

    # ueig_vec    = np.zeros(ny, dpc)
    # veig_vec    = np.zeros(ny, dpc)
    # peig_vec    = np.zeros(ny, dpc)

    # dupdx       = np.zeros(ny, dpc)
    # dupdy       = np.zeros(ny, dpc)

    # dpeig_vec_dx = np.zeros(ny, dpc)
    # dpeig_vec_dy = np.zeros(ny, dpc)


    
    # uv_eig     = ueig_vec*veig_vec
    # uv_eig_new = ueig_vec_new*veig_vec_new

    # fb = plt.figure(303)
    # plt.plot(y, uv_eig.real, 'k', label="u*v: Real part")
    # plt.plot(y, uv_eig_new.real, 'r', label="u*v (new): Real part")
    # plt.plot(y, uv_eig.imag, 'g--', label="u*v: Imag part")
    # plt.plot(y, uv_eig_new.imag, 'b--', label="u*v (new): Imag part")
    # plt.plot(y, np.abs(uv_eig), 'm', label="u*v: abs")
    # plt.plot(y, np.abs(uv_eig_new), 'c--', label="u*v (new): abs")    
    # plt.xlabel('y')
    # plt.ylabel('u*v')
    # plt.title('u*v')
    # plt.legend(loc="upper right")
    # fb.show()




    #if (not rt_flag):
        #pass
        #Re_stress     = uv_eig.real*bsfl.Up
        #Re_stress_new = uv_eig_new.real*bsfl.Up




        # if   ( mob.boussinesq == -2 ): # we need to non-dimensionalize eigenfunctions to compare with other solvers
        #     print("")
        #     print("Non-dimensionalizing eigenfunctions for R-T solver based on Chandrasekhar Eq.")
        #     ueig_vec = ueig_vec/bsfl_ref.Uref
        #     veig_vec = veig_vec/bsfl_ref.Uref
        #     weig_vec = weig_vec/bsfl_ref.Uref
        #     peig_vec = peig_vec/(bsfl_ref.rhoref*bsfl_ref.Uref**2.)
        #     reig_vec = reig_vec/bsfl_ref.rhoref


    # ueig_conj_vec = eigvects[0*ny:1*ny, idx_tar2]
    # veig_conj_vec = eigvects[1*ny:2*ny, idx_tar2]
    # weig_conj_vec = eigvects[2*ny:3*ny, idx_tar2]
    # peig_conj_vec = eigvects[3*ny:4*ny, idx_tar2]


    #ueig_conj_vec = ueig_conj_vec/norm_s_conj 
    #veig_conj_vec = veig_conj_vec/norm_s_conj
    #weig_conj_vec = weig_conj_vec/norm_s_conj 
    #peig_conj_vec = peig_conj_vec/norm_s_conj

    #ueig_real = 0.5*( ueig_vec + ueig_conj_vec )
    #ueig_imag = 0.5*( ueig_vec - ueig_conj_vec )



#    if (not rt_flag):
#        pass
        #v_from_pres = dpeig_vec_dy/( 1j*target1 -1j*alpha*bsfl.U )
        #u_from_pres = -( 1j*alpha*peig_vec + bsfl.Up*v_from_pres )/( 1j*alpha*bsfl.U - 1j*target1 )
    

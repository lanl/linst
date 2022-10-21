
import sys
import math
import warnings
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import module_utilities as mod_util

from collections import Counter

# Set automatically font size to 14 and font to "serif"
plt.rcParams['font.size'] = '14'
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
    max_tmp = 0.
    
    for i in range(0, len(eigvals)):
        if ( eigvals[i].imag > max_tmp ):
            max_tmp = eigvals[i].imag
            idx = i

    return idx


def get_idx_of_closest_eigenvalue(eigvals, abs_target, target):
    """
    This function finds the index in the eigenvalues array closest to a target eigenvalue
    """

    found = False
    
    indx_imag = -666
    indx_real = -777
    indx_abs  = -888

    diff_array_imag = np.abs( eigvals.imag - target.imag )
    diff_array_real = np.abs( eigvals.real - target.real )
    diff_array_abs  = np.abs( eigvals - target )

    sort_index_imag = np.argsort(diff_array_imag)
    sort_index_real = np.argsort(diff_array_real)
    sort_index_abs  = np.argsort(diff_array_abs)

    idx_i = sort_index_imag[0]
    idx_r = sort_index_real[0]
    idx_a = sort_index_abs[0]

    List = [idx_i, idx_r, idx_a]

    if ( idx_i != idx_r and idx_i != idx_a and idx_r != idx_a ):
        warnings.warn("Ambiguous situation ==> no eigenvalue close to target!!!!!")
        print("No reliable value found for idx_found ==> setting idx_found to 1")
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
    datafile_path = "./eigenvalues_ny" + str(ny) + ".dat" 
    np.savetxt(datafile_path , data_out, fmt=['%21.11e','%21.11e'])

def plot_cheb_baseflow(ny, y, yi, U, Up):

    ptn = plt.gcf().number
    
    #plt.rc('text', usetex=True)
    #plt.rcParams["mathtext.fontset"]
    
    f = plt.figure(ptn)
    plt.plot(np.zeros(ny), y, 'bs', markerfacecolor='none', label="Grid pts")
    plt.plot(np.zeros(ny), yi, 'r+', label=" Chebyshev pts")
    plt.xlabel('')
    plt.ylabel('y/yi')
    plt.title('Points distributions')
    plt.legend(loc="upper right")
    f.show()

    ptn=ptn+1
    
    f = plt.figure(ptn)
    plt.plot(U, y, 'k', markerfacecolor='none', label="U [-]", linewidth=1.5)
    plt.xlabel(r'$\bar{U}$', fontsize=20)
    plt.xlim([0, 1])
    #plt.ylim([-10, 10])
    plt.ylabel('y', fontsize=20)
    plt.gcf().subplots_adjust(left=0.16)
    plt.gcf().subplots_adjust(bottom=0.13)
    #plt.title('Baseflow U-velocity profile')
    #plt.legend(loc="upper right")
    f.show()

    ptn=ptn+1

    f = plt.figure(ptn)
    plt.plot(Up, y, 'k', markerfacecolor='none', label="U' [-]", linewidth=1.5)
    plt.xlabel(r"$\bar{U}$'", fontsize=20)
    plt.xlim([-2, 2])
    #plt.ylim([-10, 10])    
    plt.ylabel('y', fontsize=20)
    plt.gcf().subplots_adjust(left=0.16)
    plt.gcf().subplots_adjust(bottom=0.13)
    #plt.title("Baseflow U' profile")
    #plt.legend(loc="upper right")
    f.show()

    input()

def plot_eigvals(eigvals):

    ptn = plt.gcf().number + 1
    
    f = plt.figure(ptn)
    plt.plot(eigvals.real, eigvals.imag, 'ks', markerfacecolor='none')
    plt.xlabel(r'$\omega_r$', fontsize=20)
    plt.ylabel(r'$\omega_i$', fontsize=20)
    plt.gcf().subplots_adjust(left=0.18)
    plt.gcf().subplots_adjust(bottom=0.13)
    #plt.title('Eigenvalue spectrum')
    #plt.xlim([-1, 1])
    #plt.ylim([-1, 1])
    f.show()

def plot_phase(phase, str_var, y):

    ptn = plt.gcf().number + 1
    
    f1 = plt.figure(ptn)
    plt.plot(phase, y, 'b')
    plt.xlabel('phase(' + str_var + ')')
    plt.ylabel('y')
    plt.gcf().subplots_adjust(left=0.16)
    plt.gcf().subplots_adjust(bottom=0.13)

    f1.show()

def plot_real_imag_part(eig_fct, str_var, y):

    ptn = plt.gcf().number + 1

    mv = 20
    
    f1 = plt.figure(ptn)
    plt.plot(eig_fct.real, y, 'k', linewidth=1.5)
    if str_var == "u":
        plt.xlabel(r'$\hat{u}_r$', fontsize=20)
    elif str_var == "v":
        plt.xlabel(r'$\hat{v}_r$', fontsize=20)
    elif str_var == "w":
        plt.xlabel(r'$\hat{w}_r$', fontsize=20)
    elif str_var == "p":
        plt.xlabel(r'$\hat{p}_r$', fontsize=20)
    else:
        plt.xlabel("real(" +str_var + ")", fontsize=20)
        print("Not an expected string!!!!!")
    plt.ylabel('y', fontsize=20)
    plt.gcf().subplots_adjust(left=0.16)
    plt.gcf().subplots_adjust(bottom=0.16)
    #plt.xlim([-1, 1])
    plt.ylim([-mv, mv])
    f1.show()

    f2 = plt.figure(ptn+1)
    plt.plot(eig_fct.imag, y, 'k', linewidth=1.5)
    if str_var == "u":
        plt.xlabel(r'$\hat{u}_i$', fontsize=20)
    elif str_var == "v":
        plt.xlabel(r'$\hat{v}_i$', fontsize=20)
    elif str_var == "w":
        plt.xlabel(r'$\hat{w}_i$', fontsize=20)
    elif str_var == "p":
        plt.xlabel(r'$\hat{p}_i$', fontsize=20)
    else:
        plt.xlabel("imag(" +str_var + ")", fontsize=20)
        print("Not an expected string!!!!!")
    plt.ylabel('y', fontsize=20)
    plt.gcf().subplots_adjust(left=0.16)
    plt.gcf().subplots_adjust(bottom=0.16)
    #plt.xlim([-1, 1])
    plt.ylim([-mv, mv])
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
    plt.ylim([-mv, mv])
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

    ratio = 1.0
    xleft, xright = ax.get_xlim()
    ybottom, ytop = ax.get_ylim()
    ax.set_aspect(abs((xright-xleft)/(ybottom-ytop))*ratio)

    plt.savefig("Plane_Poiseuille_Current",bbox_inches='tight')
    
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

def get_plot_eigvcts(ny, eigvects, target1, idx_tar1, idx_tar2, alpha, map, bsfl, plot_eigvcts):

    y  = map.y
    D1 = map.D1
    
    ueig_vec    = eigvects[0*ny:1*ny, idx_tar1]
    veig_vec    = eigvects[1*ny:2*ny, idx_tar1]
    weig_vec    = eigvects[2*ny:3*ny, idx_tar1]
    peig_vec    = eigvects[3*ny:4*ny, idx_tar1]

    ueig_conj_vec = eigvects[0*ny:1*ny, idx_tar2]
    veig_conj_vec = eigvects[1*ny:2*ny, idx_tar2]
    weig_conj_vec = eigvects[2*ny:3*ny, idx_tar2]
    peig_conj_vec = eigvects[3*ny:4*ny, idx_tar2]

    # normalizing u and v by max(abs(u))
    norm_u      = np.max(np.abs(ueig_vec))
    norm_u_conj = np.max(np.abs(ueig_conj_vec))
    
    ueig_vec = ueig_vec/norm_u
    veig_vec = veig_vec/norm_u
    weig_vec = weig_vec/norm_u
    peig_vec = peig_vec/norm_u

    ueig_conj_vec = ueig_conj_vec/norm_u_conj 
    veig_conj_vec = veig_conj_vec/norm_u_conj
    weig_conj_vec = weig_conj_vec/norm_u_conj 
    peig_conj_vec = peig_conj_vec/norm_u_conj

    #ueig_real = 0.5*( ueig_vec + ueig_conj_vec )
    #ueig_imag = 0.5*( ueig_vec - ueig_conj_vec )

    # Compute phase for v-eigenfunction
    mid_idx = mod_util.get_mid_idx(ny)
    
    phase_vvel = np.arctan2( veig_vec.imag, veig_vec.real)    
    phase_vvel = phase_vvel - phase_vvel[mid_idx]

    veig_vec_polar = np.abs(veig_vec)*np.exp(1j*phase_vvel)

    #print("np.max(np.abs(veig_vec-veig_vec_polar)) = ", np.max(np.abs(veig_vec-veig_vec_polar)))

    dpeig_vec_dx = 1j*alpha*peig_vec 
    dpeig_vec_dy = np.matmul(D1, peig_vec)

    v_from_pres = dpeig_vec_dy/( 1j*target1 -1j*alpha*bsfl.U )
    u_from_pres = -( 1j*alpha*peig_vec + bsfl.Up*v_from_pres )/( 1j*alpha*bsfl.U - 1j*target1 )

    updppdx = ueig_vec + dpeig_vec_dx
    vpdppdy = veig_vec + dpeig_vec_dy

    # Verify that continuity equation is satisfied
    dupdx   = 1j*alpha*ueig_vec
    dvpdy   = np.matmul(D1, veig_vec)

    #print("Continuity check:", np.max(np.abs(dupdx+dvpdy)))
    #print("dupdx + dvpdy = ", dupdx + dvpdy)
        
    Phi_vec      = -veig_vec/(1j*alpha)
    Phi_vec_new  = -veig_vec_polar/(1j*alpha)

    Phi_vec_new  = Phi_vec_new/1j # ADDED otherwise my real part was Michalke imaginary

    veig_vec_new = -Phi_vec_new*1j*alpha
    ueig_vec_new = np.matmul(D1, Phi_vec_new)

    dupdx_new   = 1j*alpha*ueig_vec_new
    dvpdy_new   = np.matmul(D1, veig_vec_new)

    #print("Continuity check new:", np.max(np.abs(dupdx_new+dvpdy_new)))
    #print("dupdx_new+dvpdy_new = ",dupdx_new+dvpdy_new)
    
    # normalizing u and v by max(abs(u))
    norm_u_new   = np.max(np.abs(ueig_vec_new))
    
    ueig_vec_new = ueig_vec_new/norm_u_new
    veig_vec_new = veig_vec_new/norm_u_new

    uv_eig      = ueig_vec*veig_vec
    uv_eig_new  = ueig_vec_new*veig_vec_new

    Re_stress     = uv_eig.real*bsfl.Up
    Re_stress_new = uv_eig_new.real*bsfl.Up

    ## Renormalize to match Michalke
    Phi_vec_new  = Phi_vec_new/Phi_vec_new[mid_idx]

    if (plot_eigvcts):

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

    return ueig_vec, veig_vec, weig_vec, peig_vec


def read_input_file(inFile):

    with open(inFile) as f:
        lines = f.readlines()

    nlines  = len(lines)
    #print("nlines=",nlines)

    # baseflowT: 1 -> hyperbolic-tangent baseflow, 2 -> plane Poiseuille baseflow
    i=0
    tmp = lines[i].split("#")[0]
    baseflowT = int(tmp)

    # ny
    i=i+1
    
    tmp = lines[i].split("#")[0]
    ny  = int(tmp)

    # Re
    i=i+1

    tmp = lines[i].split("#")[0]
    Re  = float(tmp)

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
    print("baseflowT = ", baseflowT)
    print("ny        = ", ny)
    print("Re        = ", Re)
    print("alpha_min = ", alpha_min)
    print("alpha_max = ", alpha_max)
    print("npts_alp  = ", npts_alp)
    print("beta      = ", beta)
    print("yinf      = ", yinf)
    print("lmap      = ", lmap)
    print("target    = ", target)
    print("")

    return baseflowT, ny, Re, alpha_min, alpha_max, npts_alp, alpha, beta, yinf, lmap, target



def compute_stream_function_check(ueig, veig, alpha, D1, y, ny):
    """
    This function compute the stream function from w-eigenfunction, then computes the u-eigenfunction from:
    u_eigenfct = D1*Phi_eigenfunction
    """

    Phi_eigfct = -veig/(1j*alpha)
    
    phase_Phi  = np.arctan2(Phi_eigfct.imag, Phi_eigfct.real)
    amp_Phi    = np.abs(Phi_eigfct)

    phase_Phi_uwrap = np.unwrap(phase_Phi)

    mid_idx   = mod_util.get_mid_idx(ny)
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

    #mod_util.plot_amplitude(ueig, "u", y)
    #mod_util.plot_amplitude(ueig_fct_check, "u (indirectly from Phi)", y)

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



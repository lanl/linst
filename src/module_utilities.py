
import warnings
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import module_utilities as mod_util

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
    abs_eigs   = np.abs(eigvals)
    diff_array = np.abs(abs_eigs-abs_target) # calculate the difference array
    
    sort_index = np.argsort(diff_array)

    tol = 1.0e-10
    found = False
    
    for i in range(0, 5):
        val = np.abs(target.imag-eigvals[sort_index[i]].imag)
        #print("val, tol: ", val, tol)
        if val < tol:
            found = True
            idx_found = sort_index[i]
            break
        else:
            if i == 4:
                print("No reliable value found for idx_found ==> setting idx_found to 1")
                idx_found = 1

    if found == True:
        print("Closest eigenvalue to target:", eigvals[idx_found])
        #print("")
    else:
        print("Could not find target")
        print("")

    return idx_found, found

    
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
    plt.ylim([-10, 10])
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
    plt.ylim([-10, 10])    
    plt.ylabel('y', fontsize=20)
    plt.gcf().subplots_adjust(left=0.16)
    plt.gcf().subplots_adjust(bottom=0.13)
    #plt.title("Baseflow U' profile")
    #plt.legend(loc="upper right")
    f.show()

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
        print("Not an expected string!!!!!")

    plt.ylabel('y', fontsize=20)
    plt.gcf().subplots_adjust(left=0.16)
    plt.gcf().subplots_adjust(bottom=0.16)
    #plt.xlim([-1, 1])
    #plt.ylim([-mv, mv])
    f1.show()
    
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
    peig_vec    = eigvects[3*ny:4*ny, idx_tar1]

    ueig_conj_vec = eigvects[0*ny:1*ny, idx_tar2]
    veig_conj_vec = eigvects[1*ny:2*ny, idx_tar2]
    peig_conj_vec = eigvects[3*ny:4*ny, idx_tar2]

    # normalizing u and v by max(abs(u))
    norm_u      = np.max(np.abs(ueig_vec))
    norm_u_conj = np.max(np.abs(ueig_conj_vec))
    
    ueig_vec = ueig_vec/norm_u
    veig_vec = veig_vec/norm_u
    peig_vec = peig_vec/norm_u

    ueig_conj_vec = ueig_conj_vec/norm_u_conj 
    veig_conj_vec = veig_conj_vec/norm_u_conj 
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

    return ueig_vec, veig_vec, peig_vec


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



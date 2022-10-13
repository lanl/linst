
import numpy as np
import matplotlib.pyplot as plt

dp  = np.dtype('d')        # double precision
dpc = np.dtype(np.cdouble) # double precision complex

i4  = np.dtype('i4') # integer 4
i8  = np.dtype('i8') # integer 8

def get_idx_of_closest_eigenvalue(eigvals, abs_target, target):
    """
    This function finds the index in the eigenvalues array closest to a target eigenvalue
    """

    abs_eigs   = np.abs(eigvals)
    diff_array = np.abs(abs_eigs-abs_target) # calculate the difference array
    
    sort_index = np.argsort(diff_array)
    
    found = False
    
    for i in range(0, 5):
        val = np.abs(target.imag-eigvals[sort_index[i]].imag)
        if val < 1.0e-12:
            found = True
            idx_found = sort_index[i]
            break
        else:
            if i == 4:
                print("No reliable value found for idx_found ==> setting idx_found to 1")
                idx_found = 1

    if found == True:
        print("Closest eigenvalue to target:", eigvals[idx_found])
    else:
        print("Could not find target")

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

    f = plt.figure(1)
    plt.plot(np.zeros(ny), y, 'bs', markerfacecolor='none', label="Grid pts")
    plt.plot(np.zeros(ny), yi, 'r+', label=" Chebyshev pts")
    plt.xlabel('')
    plt.ylabel('y/yi')
    plt.title('Points distributions')
    plt.legend(loc="upper right")
    f.show()
    
    f = plt.figure(2)
    plt.plot(U, y, 'bs', markerfacecolor='none', label="U [-]")
    plt.plot(Up, y, 'r+', markerfacecolor='none', label="dU/dy [-]")
    plt.xlabel('U; dU/dy')
    plt.ylabel('y')
    plt.title('Baseflow U and dU/dy profiles')
    plt.legend(loc="upper right")
    f.show()

    input("Press any key to continue...")

def plot_eigvals(eigvals):

    f = plt.figure(3)
    plt.plot(eigvals.real, eigvals.imag, 'b+')
    plt.xlabel('real(omega)')
    plt.ylabel('imag(omega)')
    plt.title('Eigenvalue spectrum')
    #plt.xlim([-1, 1])
    #plt.ylim([-1, 1])
    f.show()

    input("Press any key to continue...")


def plot_eigvcts(ny, eigvects, target1, idx_tar1, idx_tar2, alpha, map, bsfl):

    y  = map.y
    D1 = map.D1
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

    Phi_vec     = np.zeros(ny, dpc)
    Phi_vec_new = np.zeros(ny, dpc)

    ueig_vec    = np.zeros(ny, dpc)
    veig_vec    = np.zeros(ny, dpc)
    peig_vec    = np.zeros(ny, dpc)

    dupdx       = np.zeros(ny, dpc)
    dupdy       = np.zeros(ny, dpc)

    dpeig_vec_dx = np.zeros(ny, dpc)
    dpeig_vec_dy = np.zeros(ny, dpc)

    ueig_vec    = eigvects[0*ny:1*ny, idx_tar1]
    veig_vec    = eigvects[1*ny:2*ny, idx_tar1]
    peig_vec    = eigvects[3*ny:4*ny, idx_tar1]

    ueig_conj_vec = eigvects[0*ny:1*ny, idx_tar2]
    veig_conj_vec = eigvects[1*ny:2*ny, idx_tar2]
    peig_conj_vec = eigvects[3*ny:4*ny, idx_tar2]

    # normalizing u and v by max(abs(u))
    norm_u   = np.max(np.abs(ueig_vec))
    norm_u_conj = np.max(np.abs(ueig_conj_vec))
    
    ueig_vec = ueig_vec/norm_u
    veig_vec = veig_vec/norm_u
    peig_vec = peig_vec/norm_u

    ueig_conj_vec = ueig_conj_vec/norm_u_conj 
    veig_conj_vec = veig_conj_vec/norm_u_conj 
    peig_conj_vec = peig_conj_vec/norm_u_conj

    #ueig_real = 0.5*( ueig_vec + ueig_conj_vec )
    #ueig_imag = 0.5*( ueig_vec - ueig_conj_vec )

    fbbb = plt.figure(17)
    #plt.plot(y, ueig_real, 'k', label="u(real)")
    #plt.plot(y, ueig_imag, 'r', label="u(imag)")
    plt.plot(y, np.abs(veig_vec), 'k', label="u(real)")
    plt.plot(y, np.abs(veig_conj_vec), 'r', label="u(imag)")

    plt.xlabel('y')
    plt.ylabel('u')
    plt.title('u')
    plt.legend(loc="upper right")
    fbbb.show()
    
    # Compute phase for v-eignefunction
    nloc = ny+1
    mid_idx = int(nloc/2-1)

    phase_vvel = np.arctan2( veig_vec.imag, veig_vec.real)    
    phase_vvel = phase_vvel - phase_vvel[mid_idx]

    veig_vec_polar = np.abs(veig_vec)*np.exp(1j*phase_vvel)

    print("np.max(np.abs(veig_vec-veig_vec_polar)) = ", np.max(np.abs(veig_vec-veig_vec_polar)))

    dpeig_vec_dx = 1j*alpha*peig_vec 
    dpeig_vec_dy = np.matmul(D1, peig_vec)

    v_from_pres = dpeig_vec_dy/( 1j*target1 -1j*alpha*bsfl.U )
    u_from_pres = -( 1j*alpha*peig_vec + bsfl.Up*v_from_pres )/( 1j*alpha*bsfl.U - 1j*target1 )

    faaa = plt.figure(68)
    plt.plot(y, veig_vec.imag, 'k', label="p (imag)")
    plt.plot(y, v_from_pres.imag, 'r', label="p (imag)")
    plt.xlabel('y')
    plt.ylabel('v from pres')
    plt.title('v from pres')
    plt.legend(loc="upper right")
    faaa.show()

    updppdx = ueig_vec + dpeig_vec_dx
    vpdppdy = veig_vec + dpeig_vec_dy

    # Verify that continuity equation is satisfied
    dupdx   = 1j*alpha*ueig_vec
    dvpdy   = np.matmul(D1, veig_vec)

    print("Continuity check:", np.max(np.abs(dupdx+dvpdy)))
    #print("dupdx + dvpdy = ", dupdx + dvpdy)
        
    Phi_vec      = -veig_vec/(1j*alpha)
    Phi_vec_new  = -veig_vec_polar/(1j*alpha)

    Phi_vec_new  = Phi_vec_new/1j # ADDED otherwise my real part was Michalke imaginary

    veig_vec_new = -Phi_vec_new*1j*alpha
    ueig_vec_new = np.matmul(D1, Phi_vec_new)

    dupdx_new   = 1j*alpha*ueig_vec_new
    dvpdy_new   = np.matmul(D1, veig_vec_new)

    print("Continuity check new:", np.max(np.abs(dupdx_new+dvpdy_new)))
    #print("dupdx_new+dvpdy_new = ",dupdx_new+dvpdy_new)
    
    # normalizing u and v by max(abs(u))
    norm_u_new   = np.max(np.abs(ueig_vec_new))
    
    ueig_vec_new = ueig_vec_new/norm_u_new
    veig_vec_new = veig_vec_new/norm_u_new

    uv_eig      = ueig_vec*veig_vec
    uv_eig_new  = ueig_vec_new*veig_vec_new

    Re_stress     = uv_eig.real*bsfl.Up
    Re_stress_new = uv_eig_new.real*bsfl.Up

    fac = plt.figure(71)
    #plt.plot(y, dvpdy_new.real, 'k', label="p (real)")
    #plt.plot(y, dvpdy_new.imag, 'r', label="p (imag)")
    plt.plot(y, updppdx.real + vpdppdy.real, 'k', label="p (real)")
    plt.plot(y, updppdx.imag + vpdppdy.imag, 'r', label="p (imag)")
    plt.xlabel('y')
    plt.ylabel('u''dp''/dx + v''dp''/dy')
    plt.title('u''dp''/dx + v''dp''/dy')
    plt.legend(loc="upper right")
    fac.show()

    
    fab = plt.figure(51)
    plt.plot(y, Re_stress, 'k', label="Re stress")
    plt.plot(y, Re_stress_new, 'r', label="Imaginary part")
    #
    plt.xlabel('y')
    plt.ylabel('Reynold''s stress')
    plt.title('Reynold''s stress')
    plt.legend(loc="upper right")
    fab.show()
        
    fa = plt.figure(202)
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

    faa = plt.figure(205)
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


    
    ## Renormalize to match Michalke
    Phi_vec_new  = Phi_vec_new/Phi_vec_new[mid_idx]

    ff = plt.figure(203)
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

    input("Press any key to continue")





















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

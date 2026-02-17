# src/approach2_inverse_pinn/beam_theories2.py
# modified version of the original

# provides forward physics solvers (analytical/closed-form) for a vibrating beam:
# using only euler beam
import numpy as np

# generates a sine signal whose frequency increases with time
def r(f_amp_min, f_amp_max, f_min, f_max, t, duration):
    omega_min = 2 * np.pi * f_min
    omega_max = 2 * np.pi * f_max
    omega = (omega_min + (omega_max - omega_min) / (2 * duration) * t)
    f_amp = (f_amp_max - f_amp_min) / (duration) * t
    return f_amp * np.sin(omega * t)

def Euler_beam(f, l, b, h, rho, E0, eta, F_tip, m_tip=0.0, k_phi=0.0, k_x=0.0, xl=0.0):
    """
    Physical boundary setup (your clarified design):
      - Springs at x=0:
          EI w''(0) - k_phi w'(0) = 0
          EI w'''(0) - k_x   w(0) = 0
      - Free end at x=L with tip mass + applied force:
          EI w''(L) = 0
          EI w'''(L) - m_tip*omega^2*w(L) = F_tip

    Returns:
      d       : complex velocity response at x=xl
      d_ideal : complex velocity response at x=xl for the IDEAL case (k_x=k_phi=0 and m_tip=0)
      w_field : array (1000,2): [x_vec, w_ideal(x)]  (complex in second col)
    """
    import numpy as np

    E = E0 * (1 + 1j * eta)
    I = b * h**3 / 12
    A = b * h
    omega = 2 * np.pi * f

    kappa = (omega**2 * rho * A / (E * I))**0.25

    lam_1 = kappa
    lam_2 = 1j * kappa
    lam_3 = -kappa
    lam_4 = -1j * kappa

    # Convenience
    lam = np.array([lam_1, lam_2, lam_3, lam_4], dtype=complex)

    # Build boundary matrix for coefficients c1..c4 in w(x)=sum ci exp(lam_i x)
    # We write BCs directly in terms of w, w', w'', w'''
    #
    # w(0)      = sum ci
    # w'(0)     = sum ci*lam_i
    # w''(0)    = sum ci*lam_i^2
    # w'''(0)   = sum ci*lam_i^3
    # w(L)      = sum ci*exp(lam_i L)
    # w''(L)    = sum ci*lam_i^2*exp(lam_i L)
    # w'''(L)   = sum ci*lam_i^3*exp(lam_i L)
    #
    # BCs:
    # 1) EI w''(0) - k_phi w'(0) = 0
    # 2) EI w'''(0) - k_x  w(0)  = 0
    # 3) EI w''(L) = 0
    # 4) EI w'''(L) - m_tip*omega^2*w(L) = F_tip
    #
    # Divide equations by (EI) to keep conditioning consistent:
    # 1) w''(0) - (k_phi/(EI)) w'(0) = 0
    # 2) w'''(0) - (k_x/(EI)) w(0)   = 0
    # 3) w''(L) = 0
    # 4) w'''(L) - (m_tip*omega^2/(EI)) w(L) = F_tip/(EI)

    EI = E * I
    kphi_bar = k_phi / EI
    kx_bar   = k_x   / EI
    mbar     = (m_tip * omega**2) / EI

    expL = np.exp(lam * l)

    M = np.array([
        (lam**2) - kphi_bar * (lam),               # eq1 at x=0
        (lam**3) - kx_bar   * (1.0),               # eq2 at x=0
        (lam**2) * expL,                           # eq3 at x=L
        (lam**3) * expL - mbar * (1.0 * expL),     # eq4 at x=L
    ], dtype=complex)

    a = np.array([0.0, 0.0, 0.0, (F_tip / EI)], dtype=complex)

    c = np.linalg.solve(M, a)

    # Response at xl (displacement), then multiply by i*omega to get velocity
    w_xl = (c[0]*np.exp(lam_1*xl) + c[1]*np.exp(lam_2*xl) + c[2]*np.exp(lam_3*xl) + c[3]*np.exp(lam_4*xl))
    d = w_xl * (1j * omega)

    # Ideal case: k_x=k_phi=0, m_tip=0
    kphi_bar0 = 0.0
    kx_bar0   = 0.0
    mbar0     = 0.0

    M_ideal = np.array([
        (lam**2) - kphi_bar0 * (lam),
        (lam**3) - kx_bar0   * (1.0),
        (lam**2) * expL,
        (lam**3) * expL - mbar0 * (1.0 * expL),
    ], dtype=complex)

    a_ideal = np.array([0.0, 0.0, 0.0, (F_tip / EI)], dtype=complex)
    c_ideal = np.linalg.solve(M_ideal, a_ideal)

    w_xl_ideal = (c_ideal[0]*np.exp(lam_1*xl) + c_ideal[1]*np.exp(lam_2*xl) + c_ideal[2]*np.exp(lam_3*xl) + c_ideal[3]*np.exp(lam_4*xl))
    d_ideal = w_xl_ideal * (1j * omega)

    x_vec = np.linspace(0, l, 1000)
    w_ideal = (c_ideal[0]*np.exp(lam_1*x_vec) + c_ideal[1]*np.exp(lam_2*x_vec) + c_ideal[2]*np.exp(lam_3*x_vec) + c_ideal[3]*np.exp(lam_4*x_vec))
    w_field = np.array([x_vec, w_ideal]).T

    return d, d_ideal, w_field


def TS_Beam(b, h, l, rho, fr_vec, E0, eta, F, nu, param, m=0):
    # unchanged from your code (kept as-is; still returns magnitude only)
    k_phi, k_x = param
    I = b * h ** 3 / 12
    A = b * h
    omega_err = 2 * np.pi * fr_vec
    E = E0*(1+1j*eta)

    G = E / (2 * (1 + nu))
    kappa = 10 * (1 + nu) / (12 + 11 * nu)
    mu = E / (G * kappa)
    k = np.sqrt(I / A)

    lam = (omega_err ** 2 * rho / (E * I / A)) ** 0.25

    alpha = 0.5 * k ** 2 * lam ** 2
    part_1 = alpha ** 2 * (1 - mu)**2 + 1
    part_2 = -alpha * (1 + mu)
    lam_1 = lam * np.sqrt(np.array(part_2,dtype=complex) + np.sqrt(np.array(part_1,dtype=complex)))
    lam_2 = -lam * np.sqrt(np.array(part_2,dtype=complex) + np.sqrt(np.array(part_1,dtype=complex)))
    lam_3 = lam * np.sqrt(np.array(part_2,dtype=complex) - np.sqrt(np.array(part_1,dtype=complex)))
    lam_4 = -lam * np.sqrt(np.array(part_2,dtype=complex) - np.sqrt(np.array(part_1,dtype=complex)))

    psi_1 = np.array(1 / lam_1 * (-lam_1 ** 2 - omega_err ** 2 * rho / (kappa * G)),dtype=complex)
    psi_2 = np.array(1 / lam_2 * (-lam_2 ** 2 - omega_err ** 2 * rho / (kappa * G)),dtype=complex)
    psi_3 = np.array(1 / lam_3 * (-lam_3 ** 2 - omega_err ** 2 * rho / (kappa * G)),dtype=complex)
    psi_4 = np.array(1 / lam_4 * (-lam_4 ** 2 - omega_err ** 2 * rho / (kappa * G)),dtype=complex)
    m_0 = omega_err ** 2 * m / (G * kappa * A)
    k_l = k_x/(G*kappa*A)
    k_phi_l = k_phi/(E*I)
    M = np.zeros((len(omega_err), 4, 4),dtype=complex)

    M[:, 0, 0] = (lam_1*psi_1)
    M[:, 0, 1] = (lam_2*psi_2)
    M[:, 0, 2] = (lam_3*psi_3)
    M[:, 0, 3] = (lam_4*psi_4)

    M[:, 1, 0] = (lam_1+psi_1-m_0)
    M[:, 1, 1] = (lam_2+psi_2-m_0)
    M[:, 1, 2] = (lam_3+psi_3-m_0)
    M[:, 1, 3] = (lam_4+psi_4-m_0)

    M[:, 2, 0] = ((lam_1+psi_1+k_l)*np.exp(lam_1*l))
    M[:, 2, 1] = ((lam_2+psi_2+k_l)*np.exp(lam_2*l))
    M[:, 2, 2] = ((lam_3+psi_3+k_l)*np.exp(lam_3*l))
    M[:, 2, 3] = ((lam_4+psi_4+k_l)*np.exp(lam_4*l))

    M[:, 3, 0] = ((psi_1*lam_1+psi_1*k_phi_l)*np.exp(lam_1*l))
    M[:, 3, 1] = ((psi_2*lam_2+psi_2*k_phi_l)*np.exp(lam_2*l))
    M[:, 3, 2] = ((psi_3*lam_3+psi_3*k_phi_l)*np.exp(lam_3*l))
    M[:, 3, 3] = ((psi_4*lam_4+psi_4*k_phi_l)*np.exp(lam_4*l))

    a = np.zeros((len(omega_err), 4, 1),dtype=complex)
    a[:,1,0] = -F/(G*kappa*A)

    d = np.zeros(len(omega_err),dtype=complex)
    for i in range(len(omega_err)):
        x = np.dot(np.linalg.inv(M[i,:,:]),(a[i,:,:])).flatten()
        d[i] = (x[0]*np.exp(lam_1[i]*0.3)+x[1]*np.exp(lam_2[i]*0.3)+x[2]*np.exp(lam_3[i]*0.3)+x[3]*np.exp(lam_4[i]*0.3))*1j*omega_err[i]
    return np.abs(d/F)

def TS_Beam_ideal(b, h, l, rho, fr_vec, E0, eta, F, nu):
    # unchanged from your code (kept as-is)
    I = b * h ** 3 / 12
    A = b * h
    omega_err = 2 * np.pi * fr_vec
    E = E0*(1+1j*eta)

    G = E / (2 * (1 + nu))
    kappa = 10 * (1 + nu) / (12 + 11 * nu)
    mu = E / (G * kappa)
    k = np.sqrt(I / A)

    lam = (omega_err ** 2 * rho / (E * I / A)) ** 0.25

    alpha = 0.5 * k ** 2 * lam ** 2
    part_1 = alpha ** 2 * (1 - mu)**2 + 1
    part_2 = -alpha * (1 + mu)
    lam_1 = lam * np.sqrt(np.array(part_2,dtype=complex) + np.sqrt(np.array(part_1,dtype=complex)))
    lam_2 = -lam * np.sqrt(np.array(part_2,dtype=complex) + np.sqrt(np.array(part_1,dtype=complex)))
    lam_3 = lam * np.sqrt(np.array(part_2,dtype=complex) - np.sqrt(np.array(part_1,dtype=complex)))
    lam_4 = -lam * np.sqrt(np.array(part_2,dtype=complex) - np.sqrt(np.array(part_1,dtype=complex)))

    psi_1 = np.array(1 / lam_1 * (-lam_1 ** 2 - omega_err ** 2 * rho / (kappa * G)),dtype=complex)
    psi_2 = np.array(1 / lam_2 * (-lam_2 ** 2 - omega_err ** 2 * rho / (kappa * G)),dtype=complex)
    psi_3 = np.array(1 / lam_3 * (-lam_3 ** 2 - omega_err ** 2 * rho / (kappa * G)),dtype=complex)
    psi_4 = np.array(1 / lam_4 * (-lam_4 ** 2 - omega_err ** 2 * rho / (kappa * G)),dtype=complex)

    M = np.zeros((len(omega_err), 4, 4),dtype=complex)

    M[:, 0, 0] = (lam_1*psi_1)
    M[:, 0, 1] = (lam_2*psi_2)
    M[:, 0, 2] = (lam_3*psi_3)
    M[:, 0, 3] = (lam_4*psi_4)

    M[:, 1, 0] = (lam_1+psi_1)
    M[:, 1, 1] = (lam_2+psi_2)
    M[:, 1, 2] = (lam_3+psi_3)
    M[:, 1, 3] = (lam_4+psi_4)

    M[:, 2, 0] = (np.exp(lam_1*l))
    M[:, 2, 1] = (np.exp(lam_2*l))
    M[:, 2, 2] = (np.exp(lam_3*l))
    M[:, 2, 3] = (np.exp(lam_4*l))

    M[:, 3, 0] = ((psi_1)*np.exp(lam_1*l))
    M[:, 3, 1] = ((psi_2)*np.exp(lam_2*l))
    M[:, 3, 2] = ((psi_3)*np.exp(lam_3*l))
    M[:, 3, 3] = ((psi_4)*np.exp(lam_4*l))

    a = np.zeros((len(omega_err), 4, 1),dtype=complex)
    a[:,1,0] = -F/(G*kappa*A)

    d = np.zeros(len(omega_err),dtype=complex)
    for i in range(len(omega_err)):
        x = np.dot(np.linalg.inv(M[i,:,:]),(a[i,:,:])).flatten()
        d[i] = (x[0]*np.exp(lam_1[i]*0.3)+x[1]*np.exp(lam_2[i]*0.3)+x[2]*np.exp(lam_3[i]*0.3)+x[3]*np.exp(lam_4[i]*0.3))*1j*omega_err[i]
    return np.abs(d/F)

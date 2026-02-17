#defines the PINN physics loss for Approach 2 in the frequency domain.
"""
1. PDE residual loss (Euler–Bernoulli ODE enforced inside the domain)
2. Boundary condition loss (fixed end + free end with tip mass/inertia)
3. Data loss (match to measured/synthetic FRF points)
"""

# src/approach2_inverse_pinn/physics.py
"""
Inverse PINN physics loss (frequency domain), consistent with:
- springs at x=0 (k_x, k_phi)
- tip mass + applied force at x=L (m_tip known, F complex trainable)
- data is complex VELOCITY at sensor location(s)

PINN outputs displacement W(x,omega) = Wr + i Wi
Velocity: V = i*omega*W
"""

import torch
import torch.nn.functional as F

def _grad(y, x):
    return torch.autograd.grad(
        y, x,
        grad_outputs=torch.ones_like(y),
        create_graph=True,
        retain_graph=True
    )[0]

def _pos(raw, eps=1e-12):
    return F.softplus(raw) + eps

def inverse_loss_material(
    model,
    pinn,
    x_c, omega_c,
    # known
    m_tip: torch.Tensor,
    # trainable (raw)
    E0_raw, eta_raw, kx_raw, kphi_raw,
    Fr_raw, Fi_raw,
    # data: (omega_d, x_d, Vd_r, Vd_i)
    v_data=None,
    # weights
    w_pde=1.0, w_bc=10.0, w_data=1.0,
):
    """
    Returns: (total_loss, parts_dict)
    """

    # ---- map parameters ----
    E0   = _pos(E0_raw)
    eta  = _pos(eta_raw)
    kx   = _pos(kx_raw)
    kphi = _pos(kphi_raw)

    # Complex EI = (a + i b)
    EI_r = (E0 * model.I)          # a
    EI_i = (E0 * eta * model.I)    # b

    rhoA = model.rho * model.A

    # ---------- PDE ----------
    Wr, Wi = pinn(x_c, omega_c)

    Wr_x = _grad(Wr, x_c)
    Wr_xx = _grad(Wr_x, x_c)
    Wr_xxx = _grad(Wr_xx, x_c)
    Wr_xxxx = _grad(Wr_xxx, x_c)

    Wi_x = _grad(Wi, x_c)
    Wi_xx = _grad(Wi_x, x_c)
    Wi_xxx = _grad(Wi_xx, x_c)
    Wi_xxxx = _grad(Wi_xxx, x_c)

    omega2 = omega_c**2

    # (a+ib)(u+iv) = (a u - b v) + i(a v + b u)
    EI_W4_r = EI_r * Wr_xxxx - EI_i * Wi_xxxx
    EI_W4_i = EI_r * Wi_xxxx + EI_i * Wr_xxxx

    r_pde_r = EI_W4_r - rhoA * omega2 * Wr
    r_pde_i = EI_W4_i - rhoA * omega2 * Wi

    pde_loss = torch.mean(r_pde_r**2 + r_pde_i**2)

    # ---------- BCs ----------
    omega_b = omega_c.detach()

    # x=0
    x0 = torch.zeros_like(omega_b, device=omega_b.device, dtype=omega_b.dtype, requires_grad=True)
    Wr0, Wi0 = pinn(x0, omega_b)

    Wr0_x = _grad(Wr0, x0)
    Wr0_xx = _grad(Wr0_x, x0)
    Wr0_xxx = _grad(Wr0_xx, x0)

    Wi0_x = _grad(Wi0, x0)
    Wi0_xx = _grad(Wi0_x, x0)
    Wi0_xxx = _grad(Wi0_xx, x0)

    # Springs at x=0:
    # EI*w''(0) - kphi*w'(0) = 0
    # EI*w'''(0) - kx*w(0)   = 0
    M0_r = (EI_r * Wr0_xx - EI_i * Wi0_xx) - kphi * Wr0_x
    M0_i = (EI_r * Wi0_xx + EI_i * Wr0_xx) - kphi * Wi0_x

    V0_r = (EI_r * Wr0_xxx - EI_i * Wi0_xxx) - kx * Wr0
    V0_i = (EI_r * Wi0_xxx + EI_i * Wr0_xxx) - kx * Wi0

    bc0 = M0_r**2 + M0_i**2 + V0_r**2 + V0_i**2

    # x=L
    xL = torch.full_like(omega_b, fill_value=model.L, device=omega_b.device,
                         dtype=omega_b.dtype, requires_grad=True)
    WrL, WiL = pinn(xL, omega_b)

    WrL_x = _grad(WrL, xL)
    WrL_xx = _grad(WrL_x, xL)
    WrL_xxx = _grad(WrL_xx, xL)

    WiL_x = _grad(WiL, xL)
    WiL_xx = _grad(WiL_x, xL)
    WiL_xxx = _grad(WiL_xx, xL)

    # Free moment at x=L: EI*w''(L)=0
    ML_r = (EI_r * WrL_xx - EI_i * WiL_xx)
    ML_i = (EI_r * WiL_xx + EI_i * WrL_xx)

    # Shear with tip mass + applied force:
    # EI*w'''(L) - m_tip*omega^2*w(L) - F = 0
    Fr = Fr_raw
    Fi = Fi_raw

    VL_r = (EI_r * WrL_xxx - EI_i * WiL_xxx) - m_tip * (omega_b**2) * WrL - Fr
    VL_i = (EI_r * WiL_xxx + EI_i * WrL_xxx) - m_tip * (omega_b**2) * WiL - Fi

    bcL = ML_r**2 + ML_i**2 + VL_r**2 + VL_i**2

    bc_loss = torch.mean(bc0 + bcL)

    # ---------- Data loss (velocity) ----------
    data_loss = torch.tensor(0.0, device=omega_c.device)
    if v_data is not None:
        omega_d, x_d, Vd_r, Vd_i = v_data
        Wr_d, Wi_d = pinn(x_d, omega_d)

        # V = i*omega*W = (-omega*Wi) + i(omega*Wr)
        Vp_r = -omega_d * Wi_d
        Vp_i =  omega_d * Wr_d

        data_loss = torch.mean((Vp_r - Vd_r)**2 + (Vp_i - Vd_i)**2)

    total = w_pde*pde_loss + w_bc*bc_loss + w_data*data_loss

    parts = {
        "pde": pde_loss.detach(),
        "bc": bc_loss.detach(),
        "data": data_loss.detach(),
        "E0": E0.detach(),
        "eta": eta.detach(),
        "kx": kx.detach(),
        "kphi": kphi.detach(),
        "Fr": Fr.detach(),
        "Fi": Fi.detach(),
    }
    return total, parts


# import torch

# def _grad(y, x):
#     return torch.autograd.grad(
#         y, x,
#         grad_outputs=torch.ones_like(y),
#         create_graph=True,
#         retain_graph=True
#     )[0]

# def frequency_domain_loss(
#     model,
#     pinn,
#     x_c, omega_c,
#     tip_mass, tip_inertia,
#     w_data=None,          # (omega_d, x_d, Wd_real, Wd_imag)
#     force_tip=None,       # callable or tensor for tip force amplitude vs omega (optional)
#     w_pde=1.0, w_bc=10.0, w_data_wt=1.0
# ):
#     """
#     Collocation:
#       x_c:     (N,1) requires_grad=True
#       omega_c: (N,1) (no grad needed)
#     Unknown parameters:
#       tip_mass, tip_inertia: trainable scalars

#     NN outputs:
#       W_real(x,omega), W_imag(x,omega)

#     PDE (for each omega):
#       EI W'''' - rho A omega^2 W = 0

#     BCs:
#       Fixed end (x=0): W=0, W'=0
#       Free end (x=L) with tip mass/inertia (one common form):
#         EI W''(L) + J_tip * omega^2 * W'(L) = 0
#         EI W'''(L) - m_tip * omega^2 * W(L) = F_tip(omega)   (if forcing at tip)
#       If no forcing is provided, we use F_tip=0.
#     """

#     EI = model.E * model.I
#     rhoA = model.rho * model.A

#     # ---------- PDE residual on collocation points ----------
#     Wr, Wi = pinn(x_c, omega_c)

#     Wr_x = _grad(Wr, x_c)
#     Wr_xx = _grad(Wr_x, x_c)
#     Wr_xxx = _grad(Wr_xx, x_c)
#     Wr_xxxx = _grad(Wr_xxx, x_c)

#     Wi_x = _grad(Wi, x_c)
#     Wi_xx = _grad(Wi_x, x_c)
#     Wi_xxx = _grad(Wi_xx, x_c)
#     Wi_xxxx = _grad(Wi_xxx, x_c)

#     r_pde_r = EI * Wr_xxxx - rhoA * (omega_c ** 2) * Wr
#     r_pde_i = EI * Wi_xxxx - rhoA * (omega_c ** 2) * Wi
#     pde_loss = torch.mean(r_pde_r**2 + r_pde_i**2)

#     # ---------- Boundary conditions ----------
#     # We enforce BCs over the set of unique omegas in omega_c by sampling boundary points at x=0 and x=L.
#     # Build boundary omega batch:
#     omega_b = omega_c.detach()  # same omegas, but we don't need grads w.r.t omega

#     # x=0 boundary
#     x0 = torch.zeros_like(omega_b, device=omega_b.device, dtype=omega_b.dtype, requires_grad=True)
#     Wr0, Wi0 = pinn(x0, omega_b)
#     Wr0_x = _grad(Wr0, x0)
#     Wi0_x = _grad(Wi0, x0)

#     bc_fixed = (Wr0**2 + Wi0**2) + (Wr0_x**2 + Wi0_x**2)

#     # x=L boundary
#     xL = torch.full_like(omega_b, fill_value=model.L, device=omega_b.device, dtype=omega_b.dtype, requires_grad=True)
#     WrL, WiL = pinn(xL, omega_b)

#     WrL_x = _grad(WrL, xL)
#     WrL_xx = _grad(WrL_x, xL)
#     WrL_xxx = _grad(WrL_xx, xL)

#     WiL_x = _grad(WiL, xL)
#     WiL_xx = _grad(WiL_x, xL)
#     WiL_xxx = _grad(WiL_xx, xL)

#     # Tip force (optional)
#     if force_tip is None:
#         F = torch.zeros_like(omega_b)
#     elif callable(force_tip):
#         F = force_tip(omega_b)
#     else:
#         # assume tensor broadcastable to omega_b
#         F = force_tip

#     # Free-end BC residuals (real/imag split; coefficients are real)
#     # Moment-like:
#     r_m_r = EI * WrL_xx + tip_inertia * (omega_b**2) * WrL_x
#     r_m_i = EI * WiL_xx + tip_inertia * (omega_b**2) * WiL_x

#     # Shear-like:
#     r_s_r = EI * WrL_xxx - tip_mass * (omega_b**2) * WrL - F
#     r_s_i = EI * WiL_xxx - tip_mass * (omega_b**2) * WiL  # forcing assumed real; if complex, split too

#     bc_free = r_m_r**2 + r_m_i**2 + r_s_r**2 + r_s_i**2
#     bc_loss = torch.mean(bc_fixed + bc_free)

#     # ---------- Data loss (anchors FRF) ----------
#     data_loss = torch.tensor(0.0, device=omega_c.device)
#     if w_data is not None:
#         omega_d, x_d, Wd_r, Wd_i = w_data
#         # ensure shapes (N,1)
#         Wr_d, Wi_d = pinn(x_d, omega_d)
#         data_loss = torch.mean((Wr_d - Wd_r)**2 + (Wi_d - Wd_i)**2)

#     total = w_pde*pde_loss + w_bc*bc_loss + w_data_wt*data_loss
#     return total, {"pde": pde_loss.detach(), "bc": bc_loss.detach(), "data": data_loss.detach()}





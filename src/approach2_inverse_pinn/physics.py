#defines the PINN physics loss for Approach 2 in the frequency domain.
"""
1. PDE residual loss (Euler–Bernoulli ODE enforced inside the domain)
2. Boundary condition loss (fixed end + free end with tip mass/inertia)
3. Data loss (match to measured/synthetic FRF points)
"""

import torch

def _grad(y, x):
    return torch.autograd.grad(
        y, x,
        grad_outputs=torch.ones_like(y),
        create_graph=True,
        retain_graph=True
    )[0]

def frequency_domain_loss(
    model,
    pinn,
    x_c, omega_c,
    tip_mass, tip_inertia,
    w_data=None,          # (omega_d, x_d, Wd_real, Wd_imag)
    force_tip=None,       # callable or tensor for tip force amplitude vs omega (optional)
    w_pde=1.0, w_bc=10.0, w_data_wt=1.0
):
    """
    Collocation:
      x_c:     (N,1) requires_grad=True
      omega_c: (N,1) (no grad needed)
    Unknown parameters:
      tip_mass, tip_inertia: trainable scalars

    NN outputs:
      W_real(x,omega), W_imag(x,omega)

    PDE (for each omega):
      EI W'''' - rho A omega^2 W = 0

    BCs:
      Fixed end (x=0): W=0, W'=0
      Free end (x=L) with tip mass/inertia (one common form):
        EI W''(L) + J_tip * omega^2 * W'(L) = 0
        EI W'''(L) - m_tip * omega^2 * W(L) = F_tip(omega)   (if forcing at tip)
      If no forcing is provided, we use F_tip=0.
    """

    EI = model.E * model.I
    rhoA = model.rho * model.A

    # ---------- PDE residual on collocation points ----------
    Wr, Wi = pinn(x_c, omega_c)

    Wr_x = _grad(Wr, x_c)
    Wr_xx = _grad(Wr_x, x_c)
    Wr_xxx = _grad(Wr_xx, x_c)
    Wr_xxxx = _grad(Wr_xxx, x_c)

    Wi_x = _grad(Wi, x_c)
    Wi_xx = _grad(Wi_x, x_c)
    Wi_xxx = _grad(Wi_xx, x_c)
    Wi_xxxx = _grad(Wi_xxx, x_c)

    r_pde_r = EI * Wr_xxxx - rhoA * (omega_c ** 2) * Wr
    r_pde_i = EI * Wi_xxxx - rhoA * (omega_c ** 2) * Wi
    pde_loss = torch.mean(r_pde_r**2 + r_pde_i**2)

    # ---------- Boundary conditions ----------
    # We enforce BCs over the set of unique omegas in omega_c by sampling boundary points at x=0 and x=L.
    # Build boundary omega batch:
    omega_b = omega_c.detach()  # same omegas, but we don't need grads w.r.t omega

    # x=0 boundary
    x0 = torch.zeros_like(omega_b, device=omega_b.device, dtype=omega_b.dtype, requires_grad=True)
    Wr0, Wi0 = pinn(x0, omega_b)
    Wr0_x = _grad(Wr0, x0)
    Wi0_x = _grad(Wi0, x0)

    bc_fixed = (Wr0**2 + Wi0**2) + (Wr0_x**2 + Wi0_x**2)

    # x=L boundary
    xL = torch.full_like(omega_b, fill_value=model.L, device=omega_b.device, dtype=omega_b.dtype, requires_grad=True)
    WrL, WiL = pinn(xL, omega_b)

    WrL_x = _grad(WrL, xL)
    WrL_xx = _grad(WrL_x, xL)
    WrL_xxx = _grad(WrL_xx, xL)

    WiL_x = _grad(WiL, xL)
    WiL_xx = _grad(WiL_x, xL)
    WiL_xxx = _grad(WiL_xx, xL)

    # Tip force (optional)
    if force_tip is None:
        F = torch.zeros_like(omega_b)
    elif callable(force_tip):
        F = force_tip(omega_b)
    else:
        # assume tensor broadcastable to omega_b
        F = force_tip

    # Free-end BC residuals (real/imag split; coefficients are real)
    # Moment-like:
    r_m_r = EI * WrL_xx + tip_inertia * (omega_b**2) * WrL_x
    r_m_i = EI * WiL_xx + tip_inertia * (omega_b**2) * WiL_x

    # Shear-like:
    r_s_r = EI * WrL_xxx - tip_mass * (omega_b**2) * WrL - F
    r_s_i = EI * WiL_xxx - tip_mass * (omega_b**2) * WiL  # forcing assumed real; if complex, split too

    bc_free = r_m_r**2 + r_m_i**2 + r_s_r**2 + r_s_i**2
    bc_loss = torch.mean(bc_fixed + bc_free)

    # ---------- Data loss (anchors FRF) ----------
    data_loss = torch.tensor(0.0, device=omega_c.device)
    if w_data is not None:
        omega_d, x_d, Wd_r, Wd_i = w_data
        # ensure shapes (N,1)
        Wr_d, Wi_d = pinn(x_d, omega_d)
        data_loss = torch.mean((Wr_d - Wd_r)**2 + (Wi_d - Wd_i)**2)

    total = w_pde*pde_loss + w_bc*bc_loss + w_data_wt*data_loss
    return total, {"pde": pde_loss.detach(), "bc": bc_loss.detach(), "data": data_loss.detach()}

# import torch

# def euler_bernoulli_loss(model, pinn, x, t, tip_mass, tip_inertia):
#     """
#     model: BeamModel instance
#     pinn: neural network predicting W(x,t)
#     x, t: torch tensors with requires_grad=True
#     tip_mass, tip_inertia: trainable torch Scalars
#     """
#     # Predict W
#     W = pinn(x, t)

#     # Compute derivatives
#     W_t = torch.autograd.grad(W, t, grad_outputs=torch.ones_like(W), create_graph=True)[0]
#     W_tt = torch.autograd.grad(W_t, t, grad_outputs=torch.ones_like(W), create_graph=True)[0]

#     W_x = torch.autograd.grad(W, x, grad_outputs=torch.ones_like(W), create_graph=True)[0]
#     W_xx = torch.autograd.grad(W_x, x, grad_outputs=torch.ones_like(W), create_graph=True)[0]
#     W_xxx = torch.autograd.grad(W_xx, x, grad_outputs=torch.ones_like(W), create_graph=True)[0]
#     W_xxxx = torch.autograd.grad(W_xxx, x, grad_outputs=torch.ones_like(W), create_graph=True)[0]

#     # PDE residual
#     f = model.rho * model.A * W_tt + model.E * model.I * W_xxxx
#     pde_loss = torch.mean(f**2)

#     # Boundary conditions at free end x=L
#     W_L = pinn(torch.tensor([[model.L]], dtype=torch.float32), torch.tensor([[0.0]], dtype=torch.float32))
#     W_x_L = torch.autograd.grad(W_L, torch.tensor([[model.L]], dtype=torch.float32), grad_outputs=torch.ones_like(W_L), create_graph=True)[0]

#     # Tip mass and inertia BCs (simplified)
#     bc_loss = (W_L)**2 + (W_x_L)**2  # placeholder, can improve using proper dynamic BC

#     return pde_loss + bc_loss



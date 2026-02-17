# src/approach2_inverse_pinn/train.py
import torch

from beam_model import BeamModel
from approach2_inverse_pinn.pinn_model import PINN
from approach2_inverse_pinn.physics import inverse_loss_material
from approach2_inverse_pinn.data_io import load_frf_dataset_csv

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---- known beam constants ----
beam = BeamModel(L=200/1000, b=10/1000, h=1/1000, rho=2700.0)
pinn = PINN().to(device)

# ---- known magnet mass at tip ----
m_tip = torch.tensor([0.01], dtype=torch.float32, device=device)  # TODO: set your real magnet mass (kg)

# ---- trainable parameters (raw) ----
E0_raw   = torch.nn.Parameter(torch.tensor([1e10], dtype=torch.float32, device=device))
eta_raw  = torch.nn.Parameter(torch.tensor([1e-2], dtype=torch.float32, device=device))
kx_raw   = torch.nn.Parameter(torch.tensor([10.0], dtype=torch.float32, device=device))
kphi_raw = torch.nn.Parameter(torch.tensor([10.0], dtype=torch.float32, device=device))

# complex force at tip (constant over frequency in this version)
Fr_raw = torch.nn.Parameter(torch.tensor([1.0], dtype=torch.float32, device=device))
Fi_raw = torch.nn.Parameter(torch.tensor([0.0], dtype=torch.float32, device=device))

optimizer = torch.optim.Adam(
    list(pinn.parameters()) + [E0_raw, eta_raw, kx_raw, kphi_raw, Fr_raw, Fi_raw],
    lr=1e-3
)

# ----- Collocation sampling (x, omega) -----
f_min_hz, f_max_hz = 1.0, 1000.0
omega_min = 2.0 * torch.pi * torch.tensor([f_min_hz], device=device)
omega_max = 2.0 * torch.pi * torch.tensor([f_max_hz], device=device)

Nx = 30
Nw = 25

x = torch.linspace(0.0, beam.L, Nx, device=device).view(-1, 1)
omega = torch.linspace(omega_min.item(), omega_max.item(), Nw, device=device).view(-1, 1)

xx, ww = torch.meshgrid(x.squeeze(), omega.squeeze(), indexing="ij")
x_c = xx.reshape(-1, 1).clone().detach().requires_grad_(True)
omega_c = ww.reshape(-1, 1).clone().detach()

# ----- Load VELOCITY data -----
CSV_PATH = "data/processed/frf_dataset.csv"
v_data = load_frf_dataset_csv(CSV_PATH, device=device)

epochs = 2000
for epoch in range(epochs):
    optimizer.zero_grad()

    loss, parts = inverse_loss_material(
        model=beam,
        pinn=pinn,
        x_c=x_c, omega_c=omega_c,
        m_tip=m_tip,
        E0_raw=E0_raw, eta_raw=eta_raw, kx_raw=kx_raw, kphi_raw=kphi_raw,
        Fr_raw=Fr_raw, Fi_raw=Fi_raw,
        v_data=v_data,
        w_pde=1.0, w_bc=10.0, w_data=1.0
    )

    loss.backward()
    optimizer.step()

    if epoch % 100 == 0:
        print(
            f"Epoch {epoch:5d} | Loss {loss.item():.3e} | "
            f"PDE {parts['pde'].item():.2e} | BC {parts['bc'].item():.2e} | Data {parts['data'].item():.2e} | "
            f"E0 {parts['E0'].item():.3e} | eta {parts['eta'].item():.3e} | "
            f"kx {parts['kx'].item():.3e} | kphi {parts['kphi'].item():.3e} | "
            f"F {parts['Fr'].item():.3e} + i{parts['Fi'].item():.3e}"
        )

print("\n=== Final identified parameters ===")
print(f"E0   = {parts['E0'].item():.6e} Pa")
print(f"eta  = {parts['eta'].item():.6e}")
print(f"kx   = {parts['kx'].item():.6e} N/m")
print(f"kphi = {parts['kphi'].item():.6e} N*m/rad")
print(f"F    = {parts['Fr'].item():.6e} + i {parts['Fi'].item():.6e}  N")



# import torch

# #from beam_model import BeamModel
# #from pinn_model import PIN
# #from physics import frequency_domain_loss

# from beam_model import BeamModel
# from approach2_inverse_pinn.pinn_model import PINN
# from approach2_inverse_pinn.physics import frequency_domain_loss

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# beam = BeamModel()
# pinn = PINN().to(device)

# # Unknown tip parameters (trainable)
# tip_mass = torch.nn.Parameter(torch.tensor([1.0], dtype=torch.float32, device=device))
# tip_inertia = torch.nn.Parameter(torch.tensor([0.01], dtype=torch.float32, device=device))

# optimizer = torch.optim.Adam(list(pinn.parameters()) + [tip_mass, tip_inertia], lr=1e-3)

# # ----- Collocation sampling (x, omega) -----
# # Choose frequency range in rad/s for training
# # If you have Hz, convert: omega = 2*pi*f
# f_min_hz, f_max_hz = 1.0, 500.0
# omega_min = 2.0 * torch.pi * torch.tensor([f_min_hz], device=device)
# omega_max = 2.0 * torch.pi * torch.tensor([f_max_hz], device=device)

# Nx = 20 # number of spatial collocation points
# Nw = 15 # number of frequency collocation points

# x = torch.linspace(0.0, beam.L, Nx, device=device).view(-1, 1)
# omega = torch.linspace(omega_min.item(), omega_max.item(), Nw, device=device).view(-1, 1)

# # Meshgrid -> collocation points
# xx, ww = torch.meshgrid(x.squeeze(), omega.squeeze(), indexing="ij")
# x_c = xx.reshape(-1, 1).clone().detach().requires_grad_(True)
# omega_c = ww.reshape(-1, 1).clone().detach()  # no grad needed

# # ----- Data placeholders -----
# # Replace this with your measured FRF data:
# # omega_d: (Nd,1), x_d: (Nd,1), Wd_r: (Nd,1), Wd_i: (Nd,1)

# w_data = None
# from approach2_inverse_pinn.data_io import load_frf_dataset_csv, unit_tip_force

# CSV_PATH = "data/processed/frf_dataset.csv"
# w_data = load_frf_dataset_csv(CSV_PATH, device=device)
# force_tip = unit_tip_force


# # Optional: unit force at tip (real)
# def force_tip(omega_batch):
#     return torch.ones_like(omega_batch)  # F=1

# epochs = 100
# for epoch in range(epochs):
#     optimizer.zero_grad()

#     loss, parts = frequency_domain_loss(
#         model=beam,
#         pinn=pinn,
#         x_c=x_c, omega_c=omega_c,
#         tip_mass=tip_mass, tip_inertia=tip_inertia,
#         w_data=w_data,
#         force_tip=force_tip,
#         w_pde=1.0, w_bc=10.0, w_data_wt=1.0
#     )

#     loss.backward()
#     optimizer.step()

#     if epoch % 10 == 0:
#         print(
#             f"Epoch {epoch:5d} | "
#             f"Loss {loss.item():.4e} | "
#             f"PDE {parts['pde'].item():.2e} | "
#             f"BC {parts['bc'].item():.2e} | "
#             f"Data {parts['data'].item():.2e} | "
#             f"m_tip {tip_mass.item():.4f} | "
#             f"J_tip {tip_inertia.item():.6f}"
#         )


# print("\n=== Final identified object parameters ===")
# print(f"Tip mass     m_tip = {tip_mass.item():.6f}")
# print(f"Tip inertia  J_tip = {tip_inertia.item():.6f}")







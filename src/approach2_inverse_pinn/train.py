# src/approach2_inverse_pinn/train.py
import torch

from src.beam_model import BeamModel
from .pinn_model import PINN
from .physics import inverse_loss_material
from .data_io import load_frf_dataset_csv
from .experiment_config import EXP


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---- known beam constants ----
beam = BeamModel(L=EXP.beam.L, b=EXP.beam.b, h=EXP.beam.h, rho=EXP.beam.rho)
pinn = PINN().to(device)

# ---- known magnet mass at tip ----
m_tip = torch.tensor([EXP.m_tip_kg], dtype=torch.float32, device=device)

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
    lr=EXP.lr
)

# ----- Load VELOCITY data FIRST (so we can infer freq range) -----

#CSV_PATH = str(EXP.paths.dataset_csv)
CSV_PATH = "/Users/akshiti/Desktop/implementation-pinns/data/processed/frf_dataset_exp.csv"

v_data = load_frf_dataset_csv(CSV_PATH, device=device)
omega_data = v_data[0]  # shape (N,1), rad/s

# ----- Infer f_min/f_max from dataset and expand bandwidth by 20% -----
two_pi = 2.0 * torch.pi
f_data = omega_data / two_pi  # Hz

f_min_hz_data = f_data.min().item()
f_max_hz_data = f_data.max().item()

band = f_max_hz_data - f_min_hz_data
center = 0.5 * (f_min_hz_data + f_max_hz_data)

new_band = 1.2 * band                 # +20% bandwidth
half_new = 0.5 * new_band             # +10% each side
f_min_hz = max(0.0, center - half_new)
f_max_hz = center + half_new

omega_min = two_pi * torch.tensor([f_min_hz], device=device)
omega_max = two_pi * torch.tensor([f_max_hz], device=device)

print(f"[Freq band] data: {f_min_hz_data:.3f}–{f_max_hz_data:.3f} Hz | "
      f"collocation: {f_min_hz:.3f}–{f_max_hz:.3f} Hz")

# ----- Collocation sampling (x, omega) -----
Nx = EXP.Nx
Nw = EXP.Nw

x = torch.linspace(0.0, beam.L, Nx, device=device).view(-1, 1)
omega = torch.linspace(omega_min.item(), omega_max.item(), Nw, device=device).view(-1, 1)

xx, ww = torch.meshgrid(x.squeeze(), omega.squeeze(), indexing="ij")
x_c = xx.reshape(-1, 1).clone().detach().requires_grad_(True)
omega_c = ww.reshape(-1, 1).clone().detach()

# ----- Training -----
epochs = EXP.epochs
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
# # src/approach2_inverse_pinn/train.py
# import torch

# from src.beam_model import BeamModel
# from .pinn_model import PINN
# from .physics import inverse_loss_material
# from .data_io import load_frf_dataset_csv
# from .experiment_config import EXP


# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # ---- known beam constants ----
# # beam = BeamModel(L=200/1000, b=10/1000, h=1/1000, rho=2700.0)
# beam = BeamModel(L=EXP.beam.L, b=EXP.beam.b, h=EXP.beam.h, rho=EXP.beam.rho)
# pinn = PINN().to(device)

# # ---- known magnet mass at tip ----
# # m_tip = torch.tensor([0.01], dtype=torch.float32, device=device)  # TODO: set your real magnet mass (kg)
# m_tip = torch.tensor([0.01], dtype=torch.float32, device=device)


# # ---- trainable parameters (raw) ----
# E0_raw   = torch.nn.Parameter(torch.tensor([1e10], dtype=torch.float32, device=device))
# eta_raw  = torch.nn.Parameter(torch.tensor([1e-2], dtype=torch.float32, device=device))
# kx_raw   = torch.nn.Parameter(torch.tensor([10.0], dtype=torch.float32, device=device))
# kphi_raw = torch.nn.Parameter(torch.tensor([10.0], dtype=torch.float32, device=device))

# # complex force at tip (constant over frequency in this version)
# Fr_raw = torch.nn.Parameter(torch.tensor([1.0], dtype=torch.float32, device=device))
# Fi_raw = torch.nn.Parameter(torch.tensor([0.0], dtype=torch.float32, device=device))

# # optimizer = torch.optim.Adam(
# #     list(pinn.parameters()) + [E0_raw, eta_raw, kx_raw, kphi_raw, Fr_raw, Fi_raw],
# #     lr=1e-3
# # )
# optimizer = torch.optim.Adam(
#     list(pinn.parameters()) + [E0_raw, eta_raw, kx_raw, kphi_raw, Fr_raw, Fi_raw],
#     lr=EXP.lr
# )

# # ----- Collocation sampling (x, omega) -----
# # f_min_hz, f_max_hz = 1.0, 1000.0
# e

# # Nx = 30
# # Nw = 25
# Nx = EXP.Nx
# Nw = EXP.Nw


# x = torch.linspace(0.0, beam.L, Nx, device=device).view(-1, 1)
# omega = torch.linspace(omega_min.item(), omega_max.item(), Nw, device=device).view(-1, 1)

# xx, ww = torch.meshgrid(x.squeeze(), omega.squeeze(), indexing="ij")
# x_c = xx.reshape(-1, 1).clone().detach().requires_grad_(True)
# omega_c = ww.reshape(-1, 1).clone().detach()

# # ----- Load VELOCITY data -----
# # CSV_PATH = "data/processed/frf_dataset.csv"
# # CSV_PATH = str(EXP.paths.dataset_csv) # this is for experimental data
# CSV_PATH = "/Users/akshiti/Desktop/implementation-pinns/data/processed/frf_dataset_exp.csv"
# v_data = load_frf_dataset_csv(CSV_PATH, device=device)

# # epochs = 2000
# epochs = EXP.epochs
# for epoch in range(epochs):
#     optimizer.zero_grad()

#     loss, parts = inverse_loss_material(
#         model=beam,
#         pinn=pinn,
#         x_c=x_c, omega_c=omega_c,
#         m_tip=m_tip,
#         E0_raw=E0_raw, eta_raw=eta_raw, kx_raw=kx_raw, kphi_raw=kphi_raw,
#         Fr_raw=Fr_raw, Fi_raw=Fi_raw,
#         v_data=v_data,
#         w_pde=1.0, w_bc=10.0, w_data=1.0
#     )

#     loss.backward()
#     optimizer.step()

#     if epoch % 100 == 0:
#         print(
#             f"Epoch {epoch:5d} | Loss {loss.item():.3e} | "
#             f"PDE {parts['pde'].item():.2e} | BC {parts['bc'].item():.2e} | Data {parts['data'].item():.2e} | "
#             f"E0 {parts['E0'].item():.3e} | eta {parts['eta'].item():.3e} | "
#             f"kx {parts['kx'].item():.3e} | kphi {parts['kphi'].item():.3e} | "
#             f"F {parts['Fr'].item():.3e} + i{parts['Fi'].item():.3e}"
#         )

# print("\n=== Final identified parameters ===")
# print(f"E0   = {parts['E0'].item():.6e} Pa")
# print(f"eta  = {parts['eta'].item():.6e}")
# print(f"kx   = {parts['kx'].item():.6e} N/m")
# print(f"kphi = {parts['kphi'].item():.6e} N*m/rad")
# print(f"F    = {parts['Fr'].item():.6e} + i {parts['Fi'].item():.6e}  N")










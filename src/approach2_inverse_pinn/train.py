import torch

#from beam_model import BeamModel
#from pinn_model import PIN
#from physics import frequency_domain_loss

from beam_model import BeamModel
from approach2_inverse_pinn.pinn_model import PINN
from approach2_inverse_pinn.physics import frequency_domain_loss

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

beam = BeamModel()
pinn = PINN().to(device)

# Unknown tip parameters (trainable)
tip_mass = torch.nn.Parameter(torch.tensor([1.0], dtype=torch.float32, device=device))
tip_inertia = torch.nn.Parameter(torch.tensor([0.01], dtype=torch.float32, device=device))

optimizer = torch.optim.Adam(list(pinn.parameters()) + [tip_mass, tip_inertia], lr=1e-3)

# ----- Collocation sampling (x, omega) -----
# Choose frequency range in rad/s for training
# If you have Hz, convert: omega = 2*pi*f
f_min_hz, f_max_hz = 1.0, 500.0
omega_min = 2.0 * torch.pi * torch.tensor([f_min_hz], device=device)
omega_max = 2.0 * torch.pi * torch.tensor([f_max_hz], device=device)

Nx = 20 # number of spatial collocation points
Nw = 15 # number of frequency collocation points

x = torch.linspace(0.0, beam.L, Nx, device=device).view(-1, 1)
omega = torch.linspace(omega_min.item(), omega_max.item(), Nw, device=device).view(-1, 1)

# Meshgrid -> collocation points
xx, ww = torch.meshgrid(x.squeeze(), omega.squeeze(), indexing="ij")
x_c = xx.reshape(-1, 1).clone().detach().requires_grad_(True)
omega_c = ww.reshape(-1, 1).clone().detach()  # no grad needed

# ----- Data placeholders -----
# Replace this with your measured FRF data:
# omega_d: (Nd,1), x_d: (Nd,1), Wd_r: (Nd,1), Wd_i: (Nd,1)

w_data = None
from approach2_inverse_pinn.data_io import load_frf_dataset_csv, unit_tip_force

CSV_PATH = "data/processed/frf_dataset.csv"
w_data = load_frf_dataset_csv(CSV_PATH, device=device)
force_tip = unit_tip_force


# Optional: unit force at tip (real)
def force_tip(omega_batch):
    return torch.ones_like(omega_batch)  # F=1

epochs = 100
for epoch in range(epochs):
    optimizer.zero_grad()

    loss, parts = frequency_domain_loss(
        model=beam,
        pinn=pinn,
        x_c=x_c, omega_c=omega_c,
        tip_mass=tip_mass, tip_inertia=tip_inertia,
        w_data=w_data,
        force_tip=force_tip,
        w_pde=1.0, w_bc=10.0, w_data_wt=1.0
    )

    loss.backward()
    optimizer.step()

    if epoch % 10 == 0:
        print(
            f"Epoch {epoch:5d} | "
            f"Loss {loss.item():.4e} | "
            f"PDE {parts['pde'].item():.2e} | "
            f"BC {parts['bc'].item():.2e} | "
            f"Data {parts['data'].item():.2e} | "
            f"m_tip {tip_mass.item():.4f} | "
            f"J_tip {tip_inertia.item():.6f}"
        )


print("\n=== Final identified object parameters ===")
print(f"Tip mass     m_tip = {tip_mass.item():.6f}")
print(f"Tip inertia  J_tip = {tip_inertia.item():.6f}")





# import torch
# from beam_model import BeamModel
# from pinn_model import PINN
# from physics import euler_bernoulli_loss

# # Device
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # Initialize model
# beam = BeamModel()
# pinn = PINN().to(device)

# # Unknown tip parameters as trainable scalars
# tip_mass = torch.nn.Parameter(torch.tensor([1.0], dtype=torch.float32, device=device))
# tip_inertia = torch.nn.Parameter(torch.tensor([0.01], dtype=torch.float32, device=device))

# # Optimizer
# optimizer = torch.optim.Adam(list(pinn.parameters()) + [tip_mass, tip_inertia], lr=1e-3)

# # Training data: collocation points in x-t domain
# x = torch.linspace(0, beam.L, 50).view(-1,1).to(device)
# t = torch.linspace(0, 1.0, 50).view(-1,1).to(device)
# x, t = torch.meshgrid(x.squeeze(), t.squeeze(), indexing='ij')
# x = x.reshape(-1,1)
# t = t.reshape(-1,1)

# # Training loop
# epochs = 5000
# for epoch in range(epochs):
#     optimizer.zero_grad()
#     loss = euler_bernoulli_loss(beam, pinn, x, t, tip_mass, tip_inertia)
#     loss.backward()
#     optimizer.step()

#     if epoch % 500 == 0:
#         print(f"Epoch {epoch}, Loss: {loss.item():.6f}, Tip Mass: {tip_mass.item():.4f}, Tip Inertia: {tip_inertia.item():.4f}")

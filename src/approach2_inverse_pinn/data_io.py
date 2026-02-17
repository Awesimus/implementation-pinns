# src/approach2_inverse_pinn/data_io.py
import pandas as pd
import torch

def load_frf_dataset_csv(
    csv_path: str,
    device: torch.device,
    x_col: str = "x",
    f_col: str = "f_hz",
    real_col: str = "W_real",
    imag_col: str = "W_imag",
):
    df = pd.read_csv(csv_path)

    x = torch.tensor(df[x_col].to_numpy(), dtype=torch.float32, device=device).view(-1, 1)
    f_hz = torch.tensor(df[f_col].to_numpy(), dtype=torch.float32, device=device).view(-1, 1)
    omega = 2.0 * torch.pi * f_hz  # rad/s

    Wr = torch.tensor(df[real_col].to_numpy(), dtype=torch.float32, device=device).view(-1, 1)
    Wi = torch.tensor(df[imag_col].to_numpy(), dtype=torch.float32, device=device).view(-1, 1)

    return (omega, x, Wr, Wi)

def unit_tip_force(omega_batch: torch.Tensor) -> torch.Tensor:
    return torch.ones_like(omega_batch)

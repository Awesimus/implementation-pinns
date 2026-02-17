# src/approach2_inverse_pinn/generate_dataset.py
import os
import numpy as np
import pandas as pd

from approach2_inverse_pinn.beam_theories2 import Euler_beam

def generate_frf_dataset_csv(
    out_csv="data/processed/frf_dataset.csv",
    # beam params (defaults = your example)
    l=200/1000, # beam length (m)
    b=10/1000,  # beam width (m)
    h=1/1000,   # beam thickness (m)
    rho=2700.0, # material density (kg/m³)
    E0=10e9,    # Young’s modulus (Pa)
    eta=0.01,   # material damping factor
    F=1.0,      # attached tip mass (kg)
    m=0.0/1000,
    k_phi=10.0, # rotational spring stiffness
    k_x=10.0,   # translational spring stiffness
    # sampling
    f_min=1.0,
    f_max=1000.0,
    n_freq=1000,
    x_points=(0.0,),          # list/tuple of sensor positions (meters) can be modified to have multiple values
    use_ideal=True,           # use d_ideal or d
    add_noise_std=0.0,        # optional Gaussian noise on complex outputs
):
    f_vec = np.linspace(f_min, f_max, n_freq)
    rows = []

    for xl in x_points:
        for f in f_vec:
            d, d_ideal, _ = Euler_beam(
                f=f, l=l, b=b, h=h, rho=rho, E0=E0, eta=eta, F=F,
                m=m, k_phi=k_phi, k_x=k_x, xl=xl
            )
            y = d_ideal if use_ideal else d  # complex velocity FRF at xl

            if add_noise_std > 0:
                y = y + (np.random.randn() + 1j*np.random.randn()) * add_noise_std

            rows.append({
                "x": float(xl),
                "f_hz": float(f),
                "W_real": float(np.real(y)),
                "W_imag": float(np.imag(y)),
                # metadata (optional but useful)
                "l": float(l), "b": float(b), "h": float(h), "rho": float(rho),
                "E0": float(E0), "eta": float(eta),
                "F": float(F),
                "m": float(m), "k_phi": float(k_phi), "k_x": float(k_x),
                "use_ideal": int(use_ideal),
            })

    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"Saved dataset with {len(rows)} rows -> {out_csv}")

if __name__ == "__main__":
    # Minimal change default: replicate your example (xl=0 only)
    generate_frf_dataset_csv(
        out_csv="data/processed/frf_dataset.csv",
        x_points=(0.0/1000,),   # your example uses xl = 0/1000
        n_freq=1000,
        f_min=1.0,
        f_max=1000.0,
        use_ideal=True,
        add_noise_std=0.0
    )

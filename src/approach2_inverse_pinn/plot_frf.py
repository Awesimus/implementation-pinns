# src/approach2_inverse_pinn/plot_frf.py
import os
import numpy as np
import matplotlib.pyplot as plt

#from approach2_inverse_pinn.beam_theories2 import Euler_beam
from .beam_theories2 import Euler_beam


def plot_single_frf(
    l=200/1000,
    b=10/1000,
    h=1/1000,
    rho=2700.0,
    E0=10e9,
    eta=0.01,
    F=1.0,
    m=0.0/1000,
    k_phi=10.0,
    k_x=10.0,
    xl=0.0/1000,
    f_min=1.0,
    f_max=1000.0,
    n=1000,
    use_ideal=True,
    save_path="outputs/H.svg",
    add_noise_like_old=True,
):
    # 

    f_vec = np.linspace(f_min, f_max, n)
    y = np.zeros(n, dtype=complex)

    for i, f in enumerate(f_vec):
        d, d_ideal, _ = Euler_beam(
            f=f, l=l, b=b, h=h, rho=rho, E0=E0, eta=eta, F=F,
            m=m, k_phi=k_phi, k_x=k_x, xl=xl
        )
        y[i] = d_ideal if use_ideal else d

    mag_db = 10 * np.log10(np.abs(y))
    if add_noise_like_old:
        mag_db = mag_db + np.random.rand(len(mag_db)) * 1.0

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    plt.plot(f_vec, mag_db)
    plt.xlabel("f in Hz")
    plt.ylabel("H in dB")
    plt.savefig(save_path, transparent=True)
    plt.show()

if __name__ == "__main__":
    plot_single_frf()

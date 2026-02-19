# src/approach2_inverse_pinn/experiment_config.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple


# Project root = .../<repo> (since this file is in src/approach2_inverse_pinn/)
PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Paths:
    dataset_csv: Path = PROJECT_ROOT / "data" / "processed" / "frf_dataset.csv"
    outputs_dir: Path = PROJECT_ROOT / "outputs"


@dataclass(frozen=True)
class Beam:
    L: float = 200 / 1000   # meters
    b: float = 10 / 1000    # meters
    h: float = 1 / 1000     # meters
    rho: float = 2700.0     # kg/m^3


@dataclass(frozen=True)
class FRF:
    f_min_hz: float = 1.0
    f_max_hz: float = 1000.0
    n_freq: int = 1000
    # default: measure at tip; you can set e.g. (0.0, L/2, L)
    x_points: Tuple[float, ...] | None = None
    #x_points: Tuple[float, ...] | None = (0.0, 0.1, 0.2) //meters



@dataclass(frozen=True)
class Experiment:
    paths: Paths = Paths()
    beam: Beam = Beam()
    frf: FRF = FRF()

    # Known experiment constants
    m_tip_kg: float = 0.01

    # Training (only “known knobs” you may want centralized)
    Nx: int = 30
    Nw: int = 25
    epochs: int = 2000
    lr: float = 1e-3


EXP = Experiment()

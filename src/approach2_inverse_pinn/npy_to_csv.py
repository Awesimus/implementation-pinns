from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]


INPUT_NPY = PROJECT_ROOT / "data" / "raw" / "Output.npy"


# Output CSV used by PINN training
OUTPUT_CSV = PROJECT_ROOT / "data" / "processed" / "frf_dataset_exp.csv"

# Sensor position in meters (tip measurement example)
X_SENSOR_M = 126.06 / 1000    # <-- change this to your real sensor location (meters)

# Optional: remove DC bin (0 Hz) and/or limit frequency range
DROP_F0 = True
F_MIN_HZ = None     # e.g. 1.0
F_MAX_HZ = None     # e.g. 1000.0


def main() -> None:
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    # Load the dict
    obj = np.load(INPUT_NPY, allow_pickle=True).item()
    freq_hz = np.asarray(obj["freq_s1"], dtype=float)
    spec = np.asarray(obj["s1_fft"])  # complex128

    if freq_hz.shape != spec.shape:
        raise ValueError(f"Shape mismatch: freq_s1 {freq_hz.shape} vs s1_fft {spec.shape}")

    # Build dataframe in the exact format your PINN loader expects
    df = pd.DataFrame(
        {
            "x": np.full_like(freq_hz, fill_value=float(X_SENSOR_M), dtype=float),
            "f_hz": freq_hz,
            "V_real": np.real(spec),
            "V_imag": np.imag(spec),
        }
    )

    # Optional filtering
    if DROP_F0:
        df = df[df["f_hz"] != 0.0]

    if F_MIN_HZ is not None:
        df = df[df["f_hz"] >= float(F_MIN_HZ)]

    if F_MAX_HZ is not None:
        df = df[df["f_hz"] <= float(F_MAX_HZ)]

    # Sort by frequency (nice to have)
    df = df.sort_values(["x", "f_hz"]).reset_index(drop=True)

    # Save
    df.to_csv(OUTPUT_CSV, index=False)

    # Print quick summary
    print("Saved:", OUTPUT_CSV)
    print("Rows:", len(df))
    print("x unique:", df["x"].unique()[:10], "(showing up to 10)")
    print("f range:", df["f_hz"].min(), "to", df["f_hz"].max(), "Hz")
    print("Columns:", list(df.columns))

    # Optional: show first rows
    print("\nPreview:")
    print(df.head(10))


if __name__ == "__main__":
    main()
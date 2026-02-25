# src/approach2_inverse_pinn/inspect_npy.py

from pathlib import Path
import numpy as np
import pandas as pd


# ------------------------------------------------------------
# 1) Set your file paths here
# ------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_NPY = PROJECT_ROOT / "data" / "raw" / "Output_1.npy"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ------------------------------------------------------------
# 2) Load the npy file
# ------------------------------------------------------------


data = np.load(INPUT_NPY, allow_pickle=True)

# print("\n=== BASIC INFO ===")
print("Type:", type(data))
print("Dtype:", getattr(data, "dtype", None))
print("Shape:", getattr(data, "shape", None))


# ------------------------------------------------------------
# 3) If it's a dict-like object
# ------------------------------------------------------------

if isinstance(data, np.ndarray) and data.dtype == object:
    print("\nDetected object array. Trying to interpret as dict...")
    data_obj = data.item()

    print("Object type:", type(data_obj))

    if isinstance(data_obj, dict):
        print("\nKEYS INSIDE FILE")
        for key in data_obj.keys():
            value = data_obj[key]
            print(f"\nKey: {key}")
            print("  Type:", type(value))
            print("  Shape:", getattr(value, "shape", None))
            print("  Dtype:", getattr(value, "dtype", None))

    else:
        print("Object is not a dict. Printing directly:")
        print(data_obj)

# ------------------------------------------------------------
# 4) If it's a normal numeric array
# ------------------------------------------------------------

elif isinstance(data, np.ndarray):
    print("\nARRAY CONTENT PREVIEW")
    print(data[:10])

# ------------------------------------------------------------
# 5) Optional: Save numeric arrays for inspection
# ------------------------------------------------------------

if isinstance(data, np.ndarray) and data.dtype != object:
    df = pd.DataFrame(data)
    out_path = OUTPUT_DIR / "npy_preview.csv"
    df.to_csv(out_path, index=False)
    print("\nSaved numeric array preview to:", out_path)

print("\nDone.")

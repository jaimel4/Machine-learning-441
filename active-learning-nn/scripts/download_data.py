"""Optional helper to download UCI datasets.
NOTE: This script requires internet access. If run in an offline environment, manually place CSVs.
"""
import os, sys

TARGETS = {
    "energy_efficiency.csv": "https://archive.ics.uci.edu/ml/machine-learning-databases/00242/ENB2012_data.xlsx",
    "airfoil.csv": "https://archive.ics.uci.edu/ml/machine-learning-databases/00291/airfoil_self_noise.dat",
}

def main():
    print("This is a placeholder. Please download the datasets manually and place as:")
    print("  data/raw/energy_efficiency.csv  (columns: features..., y)")
    print("  data/raw/airfoil.csv            (columns: features..., y)")
    print("See README for details.")
    sys.exit(0)

if __name__ == "__main__":
    main()

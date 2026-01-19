# config.py
from pathlib import Path

# Base directory of the project (the folder where this file lives)
BASE_DIR = Path(__file__).resolve().parent

# Data directory: put raw input files here
DATA_DIR = BASE_DIR / "data"

# Outputs directory: all intermediate and final outputs
OUT_DIR = BASE_DIR / "outputs"

# Create directories if they do not exist
DATA_DIR.mkdir(exist_ok=True)
OUT_DIR.mkdir(exist_ok=True)

if __name__ == "__main__":
    print("BASE_DIR:", BASE_DIR)
    print("DATA_DIR:", DATA_DIR)
    print("OUT_DIR:", OUT_DIR)

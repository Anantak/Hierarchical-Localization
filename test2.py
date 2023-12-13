import h5py
from pathlib import Path

# Debugging: Read a specific key
def read_key_from_h5(file_path, key):
    try:
        with h5py.File(file_path, 'r') as h5_file:
            if key in h5_file:
                data = h5_file[key][:]
                print(f"Data for {key}: {data}")
            else:
                print(f"Key {key} not found in file.")
    except Exception as e:
        print(f"Error reading key {key} from {file_path}: {e}")

# Replace 'out-114' with the key that caused the error
features_path = features_path = Path('/home/ubuntu/Anantak/Pipelines/Hierarchical-Localization/outputs/scancapture/feats-disk.h5')
read_key_from_h5(features_path, 'out-114')
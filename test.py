import h5py

def list_keys_in_h5_file(file_path):
    with h5py.File(file_path, 'r') as h5_file:
        return list(h5_file.keys())

features_path = '/home/ubuntu/Anantak/Pipelines/Hierarchical-Localization/outputs/scancapture/feats-disk.h5'
keys = list_keys_in_h5_file(features_path)



for i in keys:
    print(i)

# Check if 'out-23' is in the list
print("'out-99' in features file:", 'out-99' in keys)
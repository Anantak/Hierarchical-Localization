import h5py

def inspect_hdf5_file(file_path):
    with h5py.File(file_path, 'r') as file:
        for name in file:
            print(f"Group/Dataset name: {name}")
            try:
                keys = list(file[name].keys())
                print(f"Keys in this group/dataset: {keys}")
            except AttributeError:
                # It's a dataset, not a group, so no keys
                print("It's a dataset")

file_path = '/home/ubuntu/Anantak/Pipelines/Hierarchical-Localization/outputs/office/feats-disk-queries.h5'  # Update with your file path
inspect_hdf5_file(file_path)

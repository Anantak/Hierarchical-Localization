import h5py

def list_h5_contents(h5_file_path):
    with h5py.File(h5_file_path, 'r') as file:
        return list(file.keys())

file_path = './outputs/office/feats-disk-queries.h5'  # Adjust this path if necessary
contents = list_h5_contents(file_path)
print(contents)

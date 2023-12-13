import os
import h5py
from pathlib import Path

# Check if query images are in the dataset
dataset_dir = Path('datasets/office/frames')
query_images = ['out-162.jpg', 'test1.jpg', 'test2.jpg']

dataset_images = [f.name for f in dataset_dir.iterdir() if f.is_file()]
for query_img in query_images:
    if query_img in dataset_images:
        print(f"{query_img} is in the dataset")
    else:
        print(f"{query_img} is not in the dataset")

# Check for descriptors in global-feats-netvlad.h5
descriptor_file = Path('outputs/office/global-feats-netvlad.h5')

with h5py.File(descriptor_file, 'r') as file:
    keys = list(file.keys())
    for query_img in query_images:
        if query_img in keys:
            print(f"Descriptors for {query_img} are present in the H5 file")
        else:
            print(f"Descriptors for {query_img} are missing in the H5 file")

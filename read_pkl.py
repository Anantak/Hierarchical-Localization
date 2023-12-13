import pickle

# Replace 'file_path.pkl' with the path to your .pkl file
file_path = 'outputs/office/estimated_poses.txt_logs.pkl'

# Open the file in binary read mode
with open(file_path, 'rb') as file:
    # Load the contents from the file
    data = pickle.load(file)

# Now `data` holds the deserialized Python object.
# You can print it or perform other operations as needed
print(data)

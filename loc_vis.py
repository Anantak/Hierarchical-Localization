import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import pycolmap
from pyquaternion import Quaternion

# Function to convert quaternion to rotation matrix
def quaternion_to_rotmat(q):
    q = Quaternion(q)
    return q.rotation_matrix

# Path to the SFM reconstruction directory
reconstruction_path = '/home/ubuntu/Anantak/Pipelines/Hierarchical-Localization/datasets/office/frames/models/0/'

# Load the reconstruction
reconstruction = pycolmap.Reconstruction(reconstruction_path)

# Extract camera poses
sfm_positions = []
for _, image in reconstruction.images.items():
    # Convert quaternion to rotation matrix
    rotation_matrix = quaternion_to_rotmat([image.qvec[0], image.qvec[1], image.qvec[2], image.qvec[3]])
    # Compute camera position from rotation matrix and translation vector
    position = -rotation_matrix.T @ image.tvec
    sfm_positions.append(position)

# Load the estimated pose of the first query image
estimated_poses_file = '/home/ubuntu/Anantak/Pipelines/Hierarchical-Localization/outputs/office/estimated_poses.txt'
with open(estimated_poses_file, 'r') as file:
    line = next(file)  # Read only the first line for the first query image
    parts = line.strip().split()
    qvec = np.array([float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])])
    tvec = np.array([float(parts[5]), float(parts[6]), float(parts[7])])
    query_position = tvec

# Plotting
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot SFM camera trajectory
for pos in sfm_positions:
    ax.scatter(pos[0], pos[1], pos[2], color='blue', label='SFM Cameras' if pos is sfm_positions[0] else "")

# Plot the first query camera position
ax.scatter(query_position[0], query_position[1], query_position[2], color='red', label='Query Camera')

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.title('SFM Camera Trajectory and First Query Image Position')
ax.legend()
plt.show()

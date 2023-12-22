import pycolmap
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from pyquaternion import Quaternion
import cv2

def quaternion_to_rotmat(q):
    q = Quaternion(q)
    return q.rotation_matrix

def load_path(model_path):
    # Load the COLMAP model
    reconstruction = pycolmap.Reconstruction(model_path)

    # Extract the camera poses
    path = []
    for _, image in reconstruction.images.items():
        # Convert quaternion to rotation matrix
        rotation_matrix = quaternion_to_rotmat([image.qvec[0], image.qvec[1], image.qvec[2], image.qvec[3]])
        # Compute camera position from rotation matrix and translation vector
        position = -rotation_matrix.T @ image.tvec
        path.append(position)

    return path

def get_camera_pose(image_name, poses_file):
    # Read the poses file
    with open(poses_file, 'r') as f:
        lines = f.readlines()

    # Find the line that corresponds to the image name
    for line in lines:
        parts = line.split()
        if parts[0] == image_name:
            # Extract the quaternion and translation vector
            q = [float(x) for x in parts[1:5]]
            t = [float(x) for x in parts[5:8]]

            # Convert quaternion to rotation matrix
            rotation_matrix = quaternion_to_rotmat(q)

            return rotation_matrix, t

    raise ValueError(f"No pose found for image {image_name}")

import numpy as np

def project_points(points_3d, camera_params, camera_pose, num_points=1000, rotate=False):
    # Extract the camera parameters
    h,w, fx, fy, cx, cy = camera_params

    # Extract the camera pose
    R, t = camera_pose

    # Convert points_3d and t to numpy arrays
    points_3d = np.array(points_3d)
    t = np.array(t)

    # Compute the distances to the current position
    distances = np.linalg.norm(points_3d - t, axis=1)

    # Sort the points by distance
    sorted_indices = np.argsort(distances)

    # Select the next num_points points that are in front of the camera
    selected_points = []
    for i in sorted_indices:
        # Transform the point to the camera coordinate system
        X_cam = R @ points_3d[i] + t

        # Check if the point is in front of the camera
        if X_cam[2] > 0:
            selected_points.append(points_3d[i])
            if len(selected_points) == num_points:
                break

    # Initialize the list of projected points
    projected_points = []

    # Define the rotation matrices
    rot_2d = np.array([[0, -1], [1, 0]])  # 90 degrees rotation in 2D
    rot_y = np.array([[np.cos(np.pi/2), 0, np.sin(np.pi/2)], [0, 1, 0], [-np.sin(np.pi/2), 0, np.cos(np.pi/2)]])  # 90 degrees rotation about Y axis

    # For each selected point
    for X_world in selected_points:
        # Transform the point to the camera coordinate system
        X_cam = R @ X_world + t

        # Project the point to the 2D image plane
        x = fx * X_cam[0] / X_cam[2] + cx
        y = fy * X_cam[1] / X_cam[2] + cy

        if rotate:
            # Define the rotation matrix for a +135 degrees rotation in 2D
            angle = np.radians(135)
            rot_2d = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])

            # Translate the point to the origin
            point_2d = np.array([x, y]) - np.array([cx, cy])

            # Apply the rotation
            point_2d = rot_2d @ point_2d

            # Translate the point back to its original position
            point_2d += np.array([cx, cy])

            # Add the point to the list of projected points
            projected_points.append(point_2d)
        else:
            # Add the point to the list of projected points without rotation
            projected_points.append([x, y])

    return np.array(projected_points)


# Load the query image
query_image = cv2.imread('/home/ubuntu/Anantak/Pipelines/Hierarchical-Localization/datasets/testtrack/queries/out-681.jpg')

# Load the camera parameters
camera_model = 'SIMPLE_RADIAL'
camera_params = [1920, 1440, 1597.43, 1597.43, 953.97, 718.76]

# Load the recording path
recording_path = load_path('outputs/testtrack/models')

# print("First 5 points in recording_path:")
# for point in recording_path[:5]:
#     print(point)

# Define the path to the COLMAP model
model_path = 'outputs/testtrack/models'

# Get the camera pose for the query image
query_camera_pose = get_camera_pose('out-681.jpg', 'outputs/testtrack/estimated_poses.txt')

R, t = query_camera_pose
# print("Rotation matrix:")
# print(R)
# print("Translation vector:")
# print(t)
# print("Is rotation matrix orthogonal (R^T R = I)?")
# print(np.allclose(R.T @ R, np.eye(3)))
# print("Determinant of rotation matrix (should be 1 for right-handed, -1 for left-handed):")
# print(np.linalg.det(R))

# Project the 3D points onto the 2D image plane
projected_path = project_points(recording_path, camera_params, query_camera_pose)

# print(projected_path.min(axis=0))
# print(projected_path.max(axis=0))

# # Draw the 2D points on the image
# for i, point in enumerate(projected_path):
#     # Convert the coordinates to integers
#     x, y = map(int, point)
#     cv2.circle(query_image, (x, y), radius=1, color=(0, 255, 0), thickness=-1)
#     # cv2.putText(query_image, str(i), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

# output_path = 'outputs/testtrack/vis/query_image_with_path.jpg'

# Save the image
# cv2.imwrite(output_path, query_image)

# # # Display the image
# cv2.imshow('Query Image with Path', query_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D

# # Create a new figure
# fig = plt.figure()

# # Create a 2D axis
# ax = fig.add_subplot(111)

# # Get the original projected points
# original_points = project_points(recording_path, camera_params, query_camera_pose, rotate=False)

# # Get the rotated projected points
# rotated_points = project_points(recording_path, camera_params, query_camera_pose, rotate=True)

# # Plot the original points in green
# ax.scatter(original_points[:, 0], original_points[:, 1], color='green')

# # Plot the rotated points in red
# ax.scatter(rotated_points[:, 0], rotated_points[:, 1], color='red')

# plt.show()

# import matplotlib.pyplot as plt
# import numpy as np

# # Convert recording_path to a numpy array
# recording_path = np.array(recording_path)

# # 1. Plot the original SFM recording path in 3D
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(recording_path[:, 0], recording_path[:, 1], recording_path[:, 2], color='blue')
# ax.set_title('Original SFM recording path')

# # 2. Define a rotation matrix and a translation vector
# # Replace these with the actual rotation matrix and translation vector
# R = np.eye(3)  # rotation matrix
# T = np.zeros(3)  # translation vector

# # 3. Apply the rotation and translation to the SFM recording path
# transformed_path = (R @ recording_path.T).T + T

# # 4. Plot the transformed SFM recording path in 3D
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(transformed_path[:, 0], transformed_path[:, 1], transformed_path[:, 2], color='red')
# ax.set_title('Transformed SFM recording path')

# plt.show()

# # 5. Save the rotation matrix and translation vector for future use
# np.save('rotation_matrix.npy', R)
# np.save('translation_vector.npy', T)


import matplotlib.pyplot as plt
import cv2
import numpy as np

def quaternion_to_rotation_matrix(q):
    w, x, y, z = q
    return np.array([
        [1 - 2*y*y - 2*z*z, 2*x*y - 2*z*w, 2*x*z + 2*y*w],
        [2*x*y + 2*z*w, 1 - 2*x*x - 2*z*z, 2*y*z - 2*x*w],
        [2*x*z - 2*y*w, 2*y*z + 2*x*w, 1 - 2*x*x - 2*y*y]
    ])

# Load the image
image = cv2.imread('/home/ubuntu/Anantak/Pipelines/Hierarchical-Localization/datasets/testtrack/frames/out-100.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert from BGR to RGB

# Define the camera parameters for out-100
camera_matrix = np.array([[1596.532349, 0, 953.951294],
                          [0, 1596.532349, 718.774536],
                          [0, 0, 1]])
dist_coeffs = np.zeros(5)

# Convert recording_path to a numpy array and reshape it
recording_path_np = np.array(recording_path).reshape(-1, 1, 3)

# Convert quaternion to rotation vector
rotation_quaternion = np.array([.999913, 0.0097248, .00886862, 0.00114012])
rotation_matrix = quaternion_to_rotation_matrix(rotation_quaternion)
rotation_vector, _ = cv2.Rodrigues(rotation_matrix)

# Project the 3D points onto the 2D plane of the image
points_2d, _ = cv2.projectPoints(recording_path_np, np.array([1.02748, -.395701, 1.8695]), rotation_vector, camera_matrix, dist_coeffs)
print(recording_path_np)
print(camera_matrix)
print(dist_coeffs)
print(rotation_vector)

# Project the 3D points onto the 2D plane of the image
# points_2d, _ = cv2.projectPoints(recording_path_np, np.array([0.007362, 0.007192, 0.208646]), np.array([-0.081545, 0.005935, 0.001347, 0.996651]), camera_matrix, dist_coeffs)

# # Create a new figure
# fig, ax = plt.subplots()

# # Display the image
# ax.imshow(image)
# print('overlay')

# # Overlay the projected points
# ax.scatter(points_2d[:, 0, 0], points_2d[:, 0, 1], color='red')
# print('overlay')

# plt.show()

# # Create a new figure
# fig, ax = plt.subplots()

# # Display the image
# ax.imshow(image)

# # Overlay the projected points
# ax.scatter(points_2d[:, 0, 0], image.shape[0] - points_2d[:, 0, 1], color='red')

# plt.show()

# # Convert points_2d to integer for drawing
# points_2d = points_2d.astype(int)

# # Draw each point onto the image
# for point in points_2d:
#     cv2.circle(image, tuple(point[0]), radius=5, color=(255, 0, 0), thickness=-1)

# # Display the image with points
# cv2.imshow('Image with points', image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# Get the center of the image
# image_center = np.array([image.shape[1] / 2, image.shape[0] / 2])

# Define the rotation matrix for a +90 degree rotation about the x-axis
# theta = np.radians(180)
# rotation_matrix = np.array([[1, 0, 0], [0, np.cos(theta), -np.sin(theta)], [0, np.sin(theta), np.cos(theta)]])

# # Apply the rotation to the 3D points
# recording_path_np_rotated = np.dot(recording_path_np, rotation_matrix)

# Define the translation vector
translation_vector = np.zeros((3, 1))

# Define the rotation matrix for a +90 degree rotation about the y-axis
theta = np.radians(90)
rotation_matrix = np.array([[np.cos(theta), 0, np.sin(theta)], [0, 1, 0], [-np.sin(theta), 0, np.cos(theta)]])

# Apply the rotation to the 3D points
recording_path_np_rotated = np.dot(recording_path_np, rotation_matrix.T)  # Transpose the rotation matrix

# Project the rotated points to 2D
points_2d, _ = cv2.projectPoints(recording_path_np_rotated, rotation_vector, translation_vector, camera_matrix, dist_coeffs)

# Convert points_2d to integer for drawing
points_2d = points_2d.astype(int)

# Draw each point onto the image and annotate with image name
for i, point in enumerate(points_2d):
    cv2.circle(image, tuple(point[0]), radius=5, color=(255, 0, 0), thickness=-1)
    cv2.putText(image, str(i), (point[0][0], point[0][1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)


# Display the image with points
cv2.imshow('Image with points', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
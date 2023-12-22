import pycolmap
import numpy as np

# Get the camera pose estimation result
ret = {
    'tvec': np.array([0.007362, 0.007192, 0.208646]),
    'qvec': np.array([-0.081545, 0.005935, 0.001347, 0.996651])
}

# Create a pycolmap.Camera object
camera = pycolmap.Camera(
    id=1,  # ID of the camera
    model='PINHOLE',  # Camera model
    width=1920,  # Image width
    height=1440,  # Image height
    params=np.array([1596.532349, 1596.532349, 953.951294, 718.774536])  # Camera parameters
)

# Get the name of the query image
query = 'out-681.jpg'

# Load the SFM model
points3D, _ = pycolmap.read_model(path="outputs/testtrack/models", ext=".bin")

# Load the images
images = pycolmap.read_images_binary(path="datasets/testtrack/frames")

# Create a pycolmap.Model object
model = pycolmap.Model(points3D=points3D, images=images)

# Get the log from the 3D reconstruction process
# You need to replace get_log() with your actual function
log = get_log()

# Create a 3D figure
fig = viz_3d.create_figure()

# Plot the camera pose
viz_3d.plot_camera_colmap(fig, pose, camera, color='rgba(0,255,0,0.5)', name=query, fill=True)

# Get the 3D coordinates of the inlier points
inl_3d = np.array([model.points3D[pid].xyz for pid in np.array(log['points3D_ids'])[ret['inliers']]])

# Plot the 3D points
viz_3d.plot_points(fig, inl_3d, color="lime", ps=1, name=query)

# Show the figure
fig.show()
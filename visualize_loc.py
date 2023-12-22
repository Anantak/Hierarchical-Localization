import sys
import numpy as np
import pycolmap
from hloc.utils.viz_3d import init_figure, plot_reconstruction, plot_camera_colmap, plot_points
import pickle

def visualize_localization(model_path, ret, log, camera):
    # Load the reconstruction
    rec = pycolmap.Reconstruction()
    rec.read(model_path)

    # Initialize figure and plot reconstruction
    fig = init_figure()
    plot_reconstruction(fig, rec, points_rgb=True)

    # Convert the estimated pose to a pycolmap Image object and plot it
    pose = pycolmap.Image(tvec=ret['tvec'], qvec=ret['qvec'])
    plot_camera_colmap(fig, pose, camera, color='rgba(0,255,0,0.5)', name="query", fill=True)

    # Visualize 2D-3D correspondences
    inl_3d = np.array([rec.points3D[pid].xyz for pid in np.array(log['points3D_ids'])[ret['inliers']]])
    plot_points(fig, inl_3d, color="lime", ps=1, name="query")

    # Show the figure
    fig.show()

def load_localization_result(result_file_path):
    # Load the localization result from the file
    with open(result_file_path, 'rb') as f:
        localization_result = pickle.load(f)
    return localization_result

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python visualize_localization.py <path_to_model_directory> <localization_result_file>")
        sys.exit(1)

    model_path = sys.argv[1]
    result_file_path = sys.argv[2]
    ret, log, camera = load_localization_result(result_file_path)

    visualize_localization(model_path, ret, log, camera)


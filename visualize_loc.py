import sys
import numpy as np
import pycolmap
from hloc.utils.viz_3d import init_figure, plot_reconstruction, plot_camera_colmap, plot_points
import pickle
import csv
import ast
import pycolmap

def visualize_localization(model_path, ret, camera):
    # Load the reconstruction
    rec = pycolmap.Reconstruction()
    rec.read(model_path)

    # Initialize figure and plot reconstruction
    fig = init_figure()
    plot_reconstruction(fig, rec, points_rgb=True)

    # Convert the estimated pose to a pycolmap Image object and plot it
    pose = pycolmap.Image(tvec=ret['tvec'], qvec=ret['qvec'])
    plot_camera_colmap(fig, pose, camera, color='rgba(0,255,0,0.5)', name="query", fill=True)

    # Show the figure
    fig.show()


def load_localization_result(result_file_path):
    # Load the localization result from the file
    with open(result_file_path, 'rb') as f:
        localization_result = pickle.load(f)
    return localization_result


def load_localization_result_csv(result_file_path):
    with open(result_file_path, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            # Assume there's only one row in the CSV
            ret = {
                'success': row['success'] == 'True',
                'num_inliers': int(row['num_inliers']),
                'qvec': ast.literal_eval(row['qvec']),
                'tvec': ast.literal_eval(row['tvec']),
            }
            log = {'points3D_ids': [], 'inliers': []}  # Modify as needed

            # Define the camera based on your camera_config
            camera_config = {'model': 'SIMPLE_RADIAL', 'width': 1920, 'height': 1440, 'params': [1597.43, 1597.43, 953.97, 718.76]}
            camera = pycolmap.Camera(model=camera_config['model'], width=camera_config['width'], height=camera_config['height'], params=camera_config['params'])

            return ret, log, camera


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python visualize_localization.py <path_to_model_directory> <localization_result_file>")
        sys.exit(1)

    model_path = sys.argv[1]
    result_file_path = sys.argv[2]
    
    ret, _, camera = load_localization_result_csv(result_file_path)

    visualize_localization(model_path, ret, camera)


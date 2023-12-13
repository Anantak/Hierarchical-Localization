import pycolmap

def read_camera_parameters(sfm_model_path):
    # Load the reconstruction
    reconstruction = pycolmap.Reconstruction(sfm_model_path)

    # Iterate over cameras in the reconstruction
    for camera_id, camera in reconstruction.cameras.items():
        print(f"Camera ID: {camera_id}")
        print(f"  Model: {camera.model_name}")
        print(f"  Width: {camera.width}")
        print(f"  Height: {camera.height}")
        print(f"  Parameters: {camera.params}")

sfm_model_path = 'datasets/office/frames/models/0/'  # Path to your SFM model
read_camera_parameters(sfm_model_path)

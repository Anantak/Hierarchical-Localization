import argparse
from pathlib import Path
from hloc import extract_features, pairs_from_retrieval, match_features, localize_sfm, visualization
import sys
import numpy as np
import pycolmap
from hloc.utils.viz_3d import init_figure, plot_reconstruction, plot_camera_colmap, plot_points
import pickle
from pycolmap import Reconstruction, Camera, Image


def read_estimated_poses(file_path):
    poses = {}
    with open(file_path, 'r') as file:
        for line in file:
            parts = line.strip().split()
            if len(parts) == 8:  # Ensure the line has the correct format
                image_name, *pose = parts
                poses[image_name] = [float(value) for value in pose]
    return poses

def plot_query_poses(fig, poses, reconstruction):
    for image_name, pose in poses.items():
        qvec = np.array(pose[:4])
        tvec = np.array(pose[4:7])

        # Find the corresponding image in the reconstruction
        image_id = next((img_id for img_id, img in reconstruction.images.items() if img.name == image_name), None)
        if image_id is not None:
            camera_id = reconstruction.images[image_id].camera_id
            camera = reconstruction.cameras[camera_id]

            # Create a temporary Image object for plotting
            temp_image = Image()
            temp_image.qvec = qvec
            temp_image.tvec = (tvec)

            plot_camera_colmap(fig, temp_image, camera, name=image_name, color='green', fill=True)
        else:
            print(f"Image {image_name} not found in reconstruction.")


def query_processing():

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', help='Name of the dataset')
    args = parser.parse_args()

    # Paths for query processing
    base_dir = Path(f'datasets/{args.dataset}')
    output_dir = Path(f'outputs/{args.dataset}')
    db_img_dir = base_dir / 'frames'
    query_dir = base_dir / 'queries'
    reference_sfm = output_dir / 'models'
    queries = base_dir / 'query_with_intrinsics.txt'
    results = output_dir / 'estimated_poses.txt'

    # Configuration for feature extraction and matching
    matcher_conf = match_features.confs['disk+lightglue']
    local_feature_conf = extract_features.confs['disk']  # Local feature configuration
    global_feature_conf = extract_features.confs['netvlad']  # Global feature configuration

    # Extract local features for DATABASE images
    print('=================Extract local for database=================')
    extract_features.main(local_feature_conf, db_img_dir, output_dir)

    # Extract local features for QUERY images
    print('=================Extract local for queries=================')
    query_local_feature_output = 'feats-disk-queries'  # Output file name for query local features
    extract_features.main(local_feature_conf, query_dir, output_dir, feature_path=Path(output_dir, query_local_feature_output+'.h5'))

    # Extract global features for DATABASE images
    print('=================Extract global for database=================')
    db_global_feature_output = 'global-feats-netvlad'  # Output file name for database features
    extract_features.main(global_feature_conf, db_img_dir, output_dir, feature_path=Path(output_dir, db_global_feature_output+'.h5'))

    # Extract global features for QUERY images
    print('=================Extract global for queries=================')
    query_global_feature_output = 'global-feats-netvlad-queries'  # Output file name for query global features
    extract_features.main(global_feature_conf, query_dir, output_dir, feature_path=Path(output_dir, query_global_feature_output+'.h5'))

    # Find image pairs via image retrieval
    print('=================Image Pairs=================')
    query_descriptors = Path(output_dir, query_global_feature_output + '.h5')
    db_descriptors = Path(output_dir, db_global_feature_output + '.h5')
    retrieval_path = output_dir / 'pairs-query-netvlad20.txt'
    pairs_from_retrieval.main(
        descriptors=query_descriptors,
        db_descriptors=db_descriptors,
        output=str(retrieval_path), 
        num_matched=20
    )

    # Match features
    print('=================Match Features=================')
    matches_file_path = output_dir / (matcher_conf['output'] + '_matches.h5')
    match_features.main(
        conf=matcher_conf, 
        pairs=retrieval_path, 
        features=Path(output_dir, query_local_feature_output+'.h5'),  # Local features file for query images
        matches=matches_file_path
    )

    # Localize query images
    print('=================Localize=================')
    localize_sfm.main(
        reference_sfm=reference_sfm,
        queries=queries,
        retrieval=retrieval_path,
        features=Path(output_dir, query_local_feature_output+'.h5'),  # Local features file for query images
        matches=matches_file_path,
        results=results,
        ransac_thresh=12.0,
        covisibility_clustering=False,
        prepend_camera_name=False
    )

    print(results)
    print("Query images processed and localized.")

    # viz_camera = Camera(model=query_camera.model_name, width=query_camera.width, height=query_camera.height, params=query_camera.params)
    # pose = Image(tvec=ret['tvec'], qvec=ret['qvec'])
    # plot_camera_colmap(fig, pose, viz_camera, name=query_image_name, color='green', fill=True)

    # plot_reconstruction(fig, reconstruction, points_rgb=True)
    # # plot_reconstruction(fig, reconstruction, ret['tvec'], points_rgb=True)

    # fig.show()

    # Read and plot estimated posesQ
    results_path = output_dir / 'estimated_poses.txt'
    estimated_poses = read_estimated_poses(results_path)

    fig = init_figure()
    reconstruction = pycolmap.Reconstruction()
    reconstruction.read(reference_sfm)

    plot_reconstruction(fig, reconstruction, points_rgb=True)
    plot_query_poses(fig, estimated_poses, reconstruction)

    fig.show()

if __name__ == '__main__':
    query_processing()

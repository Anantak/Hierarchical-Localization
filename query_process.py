import argparse
import glob  # Moved to the top
from pathlib import Path
from hloc import extract_features, pairs_from_retrieval, match_features, localize_sfm, visualization
import pickle
import pycolmap
from hloc.localize_sfm import QueryLocalizer, pose_from_cluster
from hloc.utils.read_write_model import read_model
from pycolmap import Reconstruction, Camera, Image
import numpy as np
from hloc.utils.viz_3d import init_figure, plot_reconstruction, plot_camera_colmap, plot_points
import numpy as np
import csv

def quaternion_to_rotation_matrix(quat):
    """Convert a quaternion to a rotation matrix."""
    q = np.array(quat, dtype=np.float64, copy=True)
    n = np.dot(q, q)
    if n < np.finfo(q.dtype).eps:
        return np.identity(3)
    q *= np.sqrt(2.0 / n)
    q = np.outer(q, q)
    return np.array([
        [1.0-q[2, 2]-q[3, 3],     q[1, 2]-q[3, 0],     q[1, 3]+q[2, 0]],
        [    q[1, 2]+q[3, 0], 1.0-q[1, 1]-q[3, 3],     q[2, 3]-q[1, 0]],
        [    q[1, 3]-q[2, 0],     q[2, 3]+q[1, 0], 1.0-q[1, 1]-q[2, 2]]
    ])


def project_points(points_3d, qvec, tvec, camera):
    """ Project 3D points to 2D using a given camera pose and intrinsics. """
    # Convert quaternion to rotation matrix
    R = quaternion_to_rotation_matrix(qvec)
    # Project each 3D point
    projected = np.dot(R, points_3d.T).T + tvec
    projected = projected[:, :2] / projected[:, 2, np.newaxis]
    # Apply camera intrinsics
    # Assuming camera.params = [fx, fy, cx, cy]
    fx, fy, cx, cy = camera.params
    projected[:, 0] = projected[:, 0] * fx + cx
    projected[:, 1] = projected[:, 1] * fy + cy
    return projected


def calculate_reprojection_error(points_3d, points_2d, qvec, tvec, camera):
    """ Calculate the reprojection error. """
    projected_2d = project_points(points_3d, qvec, tvec, camera)
    differences = projected_2d - points_2d
    squared_errors = np.sum(differences**2, axis=1)
    return np.sqrt(np.mean(squared_errors))


def extract_features_for_images(config, image_dir, output_dir, feature_output_name, operation_name):
    print(f'================={operation_name}=================')
    extract_features.main(config, image_dir, output_dir, feature_path=Path(output_dir, feature_output_name))
    print(f"Output file: {Path(output_dir, feature_output_name)}")


def find_image_pairs(query_descriptors, db_descriptors, output_file, num_matched, operation_name):
    print(f'================={operation_name}=================')
    pairs_from_retrieval.main(descriptors=query_descriptors, db_descriptors=db_descriptors, output=str(output_file), num_matched=num_matched)
    print(f"Output file: {output_file}")


def match_image_features(config, pairs_file, features_file, matches_file, operation_name):
    print(f'================={operation_name}=================')
    match_features.main(conf=config, pairs=pairs_file, features=features_file, matches=matches_file)
    print(f"Output file: {matches_file}")


def process_queries(query_images, localizer, camera_config, output_dir, reconstruction, fig, ref_ids_integers, dataset_name):
    summary_data = []

    for query_image_name in query_images:
        # Create a camera object with the specified configuration
        query_camera = pycolmap.Camera(model=camera_config['model'], 
                                       width=camera_config['width'], 
                                       height=camera_config['height'], 
                                       params=camera_config['params'])

        # Call pose_from_cluster with the correct argument names
        ret, log = pose_from_cluster(
            localizer=localizer,
            qname=query_image_name,  # Changed to qname
            query_camera=query_camera,  # Changed to query_camera
            db_ids=ref_ids_integers,
            features_path=output_dir / 'feats-disk-queries.h5',
            matches_path=output_dir / 'matches-disk-lightglue_matches.h5'
        )

        if not ret['success']:
            print(f"Localization FAILED for {query_image_name}")
            continue

        if ret['success']:
            points_3d = np.array([reconstruction.points3D[pid].xyz for pid in np.array(log['points3D_ids'])[ret['inliers']]])
            points_2d = log['keypoints_query'][ret['inliers']]

            # Calculate reprojection error
            reprojection_error = calculate_reprojection_error(points_3d, points_2d, ret['qvec'], ret['tvec'], query_camera)

            # Store results
            summary_data.append({
                'image_name': query_image_name,
                'num_inliers': ret['num_inliers'],
                'success': ret['success'],
                'qvec': ret['qvec'].tolist(),
                'tvec': ret['tvec'].tolist(),
                'reprojection_error': reprojection_error
            })
            print(f"Localization success for {query_image_name}")
            print(f'Query {query_image_name} found {ret["num_inliers"]}/{len(ret["inliers"])} inlier correspondences.')
            print('==========================================================')

        viz_camera = Camera(model=query_camera.model_name, width=query_camera.width, height=query_camera.height, params=query_camera.params)
        pose = Image(tvec=ret['tvec'], qvec=ret['qvec'])
        plot_camera_colmap(fig, pose, viz_camera, name=query_image_name, color='green', fill=True)

        # inl_3d = np.array([reconstruction.points3D[pid].xyz for pid in np.array(log['points3D_ids'])[ret['inliers']]])
        # plot_points(fig, inl_3d, color='green', ps=1, name=query_image_name)

    # Sort summary_data by reprojection_error and reorder columns
    summary_data.sort(key=lambda x: x['reprojection_error'])
    csv_file_path = output_dir / f'{dataset_name}_summary.csv'
    csv_file_path.parent.mkdir(parents=True, exist_ok=True)  # Ensure the directory exists

    with open(csv_file_path, 'w', newline='') as file:
        fieldnames = ['image_name', 'success', 'reprojection_error', 'num_inliers', 'qvec', 'tvec']
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_data)


def query_processing():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', help='Name of the dataset')
    args = parser.parse_args()

    dataset_name = args.dataset 
    base_dir = Path(f'datasets/{dataset_name}')
    output_dir = Path(f'outputs/{dataset_name}')
    db_img_dir = base_dir / 'frames'
    query_dir = base_dir / 'path_queries'

    # Configuration for feature extraction and matching
    local_feature_conf = extract_features.confs['disk']
    global_feature_conf = extract_features.confs['netvlad']
    matcher_conf = match_features.confs['disk+lightglue']

    # Feature extraction
    extract_features_for_images(local_feature_conf, db_img_dir, output_dir, 'feats-disk.h5', "Extracting Local Features for Database Images")
    extract_features_for_images(local_feature_conf, query_dir, output_dir, 'feats-disk-queries.h5', "Extracting Local Features for Query Images")
    extract_features_for_images(global_feature_conf, db_img_dir, output_dir, 'global-feats-netvlad.h5', "Extracting Global Features for Database Images")
    extract_features_for_images(global_feature_conf, query_dir, output_dir, 'global-feats-netvlad-queries.h5', "Extracting Global Features for Query Images")

    # Image pairs and feature matching
    find_image_pairs(Path(output_dir, 'global-feats-netvlad-queries.h5'), Path(output_dir, 'global-feats-netvlad.h5'), output_dir / 'pairs-query-netvlad20.txt', 20, "Generating Image Pairs")
    match_image_features(matcher_conf, output_dir / 'pairs-query-netvlad20.txt', Path(output_dir, 'feats-disk-queries.h5'), output_dir / 'matches-disk-lightglue_matches.h5', "Matching Features")

    print('==========================================================')

    model_path = Path(f'outputs/{dataset_name}/models/')
    pairs_file = output_dir / 'pairs-query-netvlad20.txt'
    path_query_dir = base_dir / 'path_queries' 
    query_images = [Path(file).name for file in glob.glob(str(path_query_dir / '*.jpg'))]
    reconstruction = Reconstruction()
    reconstruction.read(model_path)

    conf = {
        'estimation': {'ransac': {'max_error': 20}},
        'refinement': {'refine_focal_length': True, 'refine_extra_params': True},
    }

    localizer = QueryLocalizer(reconstruction, conf)

    # Reading the pairs file and extracting database image names
    pairs = [line.strip().split() for line in open(pairs_file, 'r').readlines()]
    db_image_names = list(set(pair[1] for pair in pairs))

    ref_ids_integers = []
    for img_name in db_image_names:
        found_id = next((img_id for img_id, img in reconstruction.images.items() if img.name == img_name), None)
        if found_id is not None:
            ref_ids_integers.append(found_id)
        else:
            print(f"No ID found for image {img_name}")

    camera_config = {
        'model': 'SIMPLE_RADIAL',
        'width': 1920,
        'height': 1440,
        'params': [1597.43, 1597.43, 953.97, 718.76]
    }
    # 1329.069092, 1329.069092, 970.617859, 718.538879 iPhone 15 pro
    # 1597.43, 1597.43, 953.97, 718.76 iPad pro

    fig = init_figure()
    plot_reconstruction(fig, reconstruction, points_rgb=True)

    path_query_dir = base_dir / 'path_queries'
    query_images = [Path(file).name for file in glob.glob(str(path_query_dir / '*.jpg'))]

    process_queries(query_images, localizer, camera_config, output_dir, reconstruction, fig, ref_ids_integers, dataset_name)

    fig.show()


if __name__ == '__main__':
    query_processing()
import argparse
import glob
from pathlib import Path
import numpy as np
import shutil
import csv
from hloc import extract_features, pairs_from_retrieval, match_features, localize_sfm
from hloc.localize_sfm import QueryLocalizer, pose_from_cluster
from pycolmap import Reconstruction, Camera, Image
import re

def get_image_number(image_name):
    # Extract the numerical part from the image name
    match = re.search(r'(\d+)', image_name)
    return int(match.group()) if match else None


def quaternion_to_rotation_matrix(quat):
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
    R = quaternion_to_rotation_matrix(qvec)
    projected = np.dot(R, points_3d.T).T + tvec
    projected = projected[:, :2] / projected[:, 2, np.newaxis]
    fx, fy, cx, cy = camera.params
    projected[:, 0] = projected[:, 0] * fx + cx
    projected[:, 1] = projected[:, 1] * fy + cy
    return projected


def calculate_reprojection_error(points_3d, points_2d, qvec, tvec, camera):
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


def match_image_features(config, pairs_file, db_features_file, query_features_file, matches_file, operation_name):
    print(f'================={operation_name}=================')
    match_features.main(conf=config, pairs=pairs_file, features=query_features_file, matches=matches_file)
    print(f"Output file: {matches_file}")


def find_closest_waypoints(query_pose, reconstruction, num_waypoints=50):
    query_position = np.array(query_pose['tvec'])  # Access as dictionary
    waypoints = []
    for image_id, image in reconstruction.images.items():
        waypoint_position = np.array(image.tvec)
        distance = np.linalg.norm(query_position - waypoint_position)
        waypoints.append((image_id, image.name, distance))
    waypoints.sort(key=lambda x: x[2])
    return waypoints[:num_waypoints]


def calculate_relative_positions(query_pose, closest_waypoints, reconstruction):
    query_position = np.array(query_pose['tvec'])  # Access as dictionary
    relative_positions = []
    for image_id, image_name, _ in closest_waypoints:
        waypoint_position = np.array(reconstruction.images[image_id].tvec)
        relative_position = waypoint_position - query_position
        relative_positions.append((image_name, relative_position.tolist()))
    return relative_positions


def find_closest_image_id(query_pose, reconstruction):
    query_position = np.array(query_pose['tvec'])
    closest_distance = float('inf')
    closest_image_id = None

    for image_id, image in reconstruction.images.items():
        waypoint_position = np.array(image.tvec)
        distance = np.linalg.norm(query_position - waypoint_position)
        if distance < closest_distance:
            closest_distance = distance
            closest_image_id = image_id

    return closest_image_id

def calculate_sequential_positions(query_pose, reconstruction, num_images=50):
    query_position = np.array(query_pose['tvec'])
    closest_image_id = find_closest_image_id(query_pose, reconstruction)

    sequential_positions = []
    total_images = len(reconstruction.images)
    image_ids = sorted(reconstruction.images.keys())

    # Find the index of the closest image ID in the sorted list of IDs
    closest_index = image_ids.index(closest_image_id)

    # Iterate through the next 50 images starting from the closest image ID
    for i in range(closest_index, min(closest_index + num_images, total_images)):
        image_id = image_ids[i]
        image = reconstruction.images[image_id]
        waypoint_position = np.array(image.tvec)
        relative_position = waypoint_position - query_position
        sequential_positions.append((image.name, relative_position.tolist()))

    return sequential_positions



def process_query(query_image_path, localizer, camera_config, output_dir, reconstruction, dataset_name):
    summary_data = []
    query_image_name = Path(query_image_path).name
    query_camera = Camera(model=camera_config['model'], width=camera_config['width'], height=camera_config['height'], params=camera_config['params'])

    pairs_file = output_dir / 'pairs-query-netvlad20.txt'
    pairs = [line.strip().split() for line in open(pairs_file, 'r').readlines()]
    db_image_names = list(set(pair[1] for pair in pairs))
    name_to_id_map = {img.name: img_id for img_id, img in reconstruction.images.items()}
    ref_ids_integers = [name_to_id_map[name] for name in db_image_names if name in name_to_id_map]

    features_path = output_dir / 'feats-disk-queries.h5'
    matches_path = output_dir / 'matches-disk-lightglue_matches.h5'

    print(f'=================LOCALIZE=================')
    ret, log = pose_from_cluster(localizer, qname=query_image_name, query_camera=query_camera, db_ids=ref_ids_integers, features_path=features_path, matches_path=matches_path)

    if not ret['success']:
        print(f"Localization FAILED for {query_image_name}")
        return []

    query_position = np.array(ret['tvec'])

    closest_distance = float('inf')
    closest_pose_id = None
    for image_id, image in reconstruction.images.items():
        waypoint_position = np.array(image.tvec)
        distance = np.linalg.norm(query_position - waypoint_position)
        if distance < closest_distance:
            closest_distance = distance
            closest_pose_id = image_id

    closest_pose_name = reconstruction.images[closest_pose_id].name
    print(f"Closest Pose ID: {closest_pose_id}, Closest Pose Name: {closest_pose_name}")

    relative_positions = calculate_sequential_positions(ret, reconstruction, num_images=50)
    relative_positions_sorted = sorted(relative_positions, key=lambda x: get_image_number(x[0]))
    points_3d = np.array([reconstruction.points3D[pid].xyz for pid in np.array(log['points3D_ids'])[ret['inliers']]])
    points_2d = log['keypoints_query'][ret['inliers']]
    reprojection_error = calculate_reprojection_error(points_3d, points_2d, ret['qvec'], ret['tvec'], query_camera)

    summary_data.append({'image_name': query_image_name, 'num_inliers': ret['num_inliers'], 'success': ret['success'], 'qvec': ret['qvec'].tolist(), 'tvec': ret['tvec'].tolist(), 'reprojection_error': reprojection_error})

    print(f'=================RESULTS=================')
    for i in relative_positions_sorted:
        print(i)

    csv_file_path = output_dir / f'{dataset_name}_summary.csv'
    csv_file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_file_path, 'w', newline='') as file:
        fieldnames = ['image_name', 'success', 'reprojection_error', 'num_inliers', 'qvec', 'tvec']
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_data)

    return relative_positions

def query_processing(dataset_name, query_image_path):
    base_dir = Path(f'/home/ubuntu/Anantak/Pipelines/Hierarchical-Localization/datasets/{dataset_name}')
    output_dir = Path(f'/home/ubuntu/Anantak/Pipelines/Hierarchical-Localization/outputs/{dataset_name}')
    db_img_dir = base_dir / 'frames'
    query_dir = Path('/home/ubuntu/Anantak/Pipelines/Hierarchical-Localization/datasets/testtrack/path_queries/')
    query_descriptors = output_dir / 'global-feats-netvlad-queries.h5'
    db_descriptors = output_dir / 'global-feats-netvlad.h5'
    feats_disk_queries = output_dir / 'feats-disk-queries.h5'
    global_feats_netvlad_queries = output_dir / 'global-feats-netvlad-queries.h5'
    matches_disk_lightglue = output_dir / 'matches-disk-lightglue_matches.h5'
    pairs_query_netvlad = output_dir / 'pairs-query-netvlad20.txt'

    for file in [feats_disk_queries, global_feats_netvlad_queries, matches_disk_lightglue, pairs_query_netvlad]:
        if file.exists():
            file.unlink()

    local_feature_conf = extract_features.confs['disk']
    global_feature_conf = extract_features.confs['netvlad']
    matcher_conf = match_features.confs['disk+lightglue']

    extract_features_for_images(local_feature_conf, query_dir, output_dir, 'feats-disk-queries.h5', "Extracting Local Features for Query Images")
    extract_features_for_images(global_feature_conf, query_dir, output_dir, 'global-feats-netvlad-queries.h5', "Extracting Global Features for Query Images")

    if not query_descriptors.is_file():
        raise FileNotFoundError(f"Query descriptors not found: {query_descriptors}")
    if not db_descriptors.is_file():
        raise FileNotFoundError(f"DB descriptors not found: {db_descriptors}")

    find_image_pairs(query_descriptors, db_descriptors, output_dir / 'pairs-query-netvlad20.txt', 40, "Generating Image Pairs")

    db_features_file = Path('/home/ubuntu/Anantak/Pipelines/Hierarchical-Localization/outputs/testtrack/feats-disk.h5')
    query_features_file = Path('/home/ubuntu/Anantak/Pipelines/Hierarchical-Localization/outputs/testtrack/feats-disk-queries.h5')
    pairs_file = Path('/home/ubuntu/Anantak/Pipelines/Hierarchical-Localization/outputs/testtrack/pairs-query-netvlad20.txt')
    matches_file = Path('/home/ubuntu/Anantak/Pipelines/Hierarchical-Localization/outputs/testtrack/matches-disk-lightglue_matches.h5')

    match_image_features(matcher_conf, pairs_file, db_features_file, query_features_file, matches_file, "Matching Features")

    model_path = Path(f'/home/ubuntu/Anantak/Pipelines/Hierarchical-Localization/outputs/testtrack/models')
    reconstruction = Reconstruction()
    reconstruction.read(model_path)

    conf = {'estimation': {'ransac': {'max_error': 20}}, 'refinement': {'refine_focal_length': True, 'refine_extra_params': True}}
    localizer = QueryLocalizer(reconstruction, conf)

    pairs = [line.strip().split() for line in open(pairs_file, 'r').readlines()]
    db_image_names = list(set(pair[1] for pair in pairs))

    ref_ids_integers = []
    for img_name in db_image_names:
        found_id = next((img_id for img_id, img in reconstruction.images.items() if img.name == img_name), None)
        if found_id is not None:
            ref_ids_integers.append(found_id)
        else:
            print(f"No ID found for image {img_name}")

    camera_config = {'model': 'SIMPLE_RADIAL', 'width': 1920, 'height': 1440, 'params': [1597.43, 1597.43, 953.97, 718.76]}

    relative_positions = process_query(query_image_path, localizer, camera_config, output_dir, reconstruction, dataset_name)

    return relative_positions

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', help='Name of the dataset')
    parser.add_argument('--query_image_path', help='Path to the query image')
    args = parser.parse_args()

    query_processing(args.dataset, args.query_image_path)

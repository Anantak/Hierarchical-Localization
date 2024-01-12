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
from hloc.utils.viz_3d import init_figure, plot_reconstruction, plot_camera_colmap, plot_points
from cprint import *

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


def extract_features_for_images(config, image_dir, output_dir, feature_output_name, operation_name):
    cprint.info(f'================={operation_name}=================')
    extract_features.main(config, image_dir, output_dir, feature_path=Path(output_dir, feature_output_name))
    cprint.ok(f"Output file: {Path(output_dir, feature_output_name)}")


def find_image_pairs(query_descriptors, db_descriptors, output_file, num_matched, operation_name):
    cprint.info(f'================={operation_name}=================')
    pairs_from_retrieval.main(descriptors=query_descriptors, db_descriptors=db_descriptors, output=str(output_file), num_matched=num_matched)
    cprint.ok(f"Output file: {output_file}")


def match_image_features(config, pairs_file, db_features_file, query_features_file, matches_file, operation_name):
    cprint.info(f'================={operation_name}=================')
    match_features.main(conf=config, pairs=pairs_file, features=query_features_file, matches=matches_file)
    cprint.ok(f"Output file: {matches_file}")


def find_closest_image_id(query_pose, reconstruction):
    query_position = np.array(query_pose['tvec'])
    closest_distance = float('inf')
    closest_image_id = None
    reference_distances = {}
    reference_tvecs = {}

    for image_id, image in reconstruction.images.items():
        waypoint_position = np.array(image.tvec)
        distance = np.linalg.norm(query_position - waypoint_position)

        # Save distances and tvecs for reference images
        image_number = get_image_number(image.name)
        if image_number is not None and image_number % 100 == 1:
            reference_distances[image.name] = (distance, waypoint_position.tolist())

        # Find the closest image
        if distance < closest_distance:
            closest_distance = distance
            closest_image_id = image_id

    # Print query image tvec
    cprint.info("=============================================================================")
    print(f"Query Image tvec: {query_position.tolist()}")

    # Sort and print distances and tvecs for reference images in ascending order of image names
    cprint.info("========================Sorted Reference Distances and tvecs========================")
    for image_name in sorted(reference_distances.keys(), key=lambda x: int(x.split('-')[1].split('.')[0])):
        distance, tvec = reference_distances[image_name]
        print(f"{image_name}: Distance = {distance}, tvec = {tvec}")

    return closest_image_id



def calculate_sequential_positions(query_pose, reconstruction, num_images=50):
    query_position = np.array(query_pose['tvec'])
    
    # Extract image numbers and sort them numerically
    images_sorted = sorted(reconstruction.images.items(), key=lambda x: get_image_number(x[1].name))

    # Find the closest image in the sorted list
    closest_image_id = find_closest_image_id(query_pose, reconstruction)
    closest_index = next(i for i, (image_id, _) in enumerate(images_sorted) if image_id == closest_image_id)

    # Get sequential positions considering wrap-around
    sequential_positions = []
    for i in range(closest_index, closest_index + num_images):
        # Wrap around if the end of the list is reached
        index = i % len(images_sorted)
        image_id, image = images_sorted[index]
        waypoint_position = np.array(image.tvec)
        relative_position = waypoint_position - query_position
        sequential_positions.append((image.name, relative_position.tolist()))

    return closest_image_id, sequential_positions


def process_query(query_image_path, localizer, camera_config, output_dir, reconstruction, dataset_name):
    summary_data = []
    query_image_name = query_image_path
    query_camera = Camera(model=camera_config['model'], width=camera_config['width'], height=camera_config['height'], params=camera_config['params'])

    pairs_file = output_dir / 'pairs-query-netvlad20.txt'
    pairs = [line.strip().split() for line in open(pairs_file, 'r').readlines()]
    db_image_names = list(set(pair[1] for pair in pairs))
    name_to_id_map = {img.name: img_id for img_id, img in reconstruction.images.items()}
    ref_ids_integers = [name_to_id_map[name] for name in db_image_names if name in name_to_id_map]

    features_path = output_dir / 'feats-disk-queries.h5'
    matches_path = output_dir / 'matches-disk-lightglue_matches.h5'

    cprint.info(f'==================================LOCALIZE==================================')
    ret, log = pose_from_cluster(localizer, qname=query_image_name, query_camera=query_camera, db_ids=ref_ids_integers, features_path=features_path, matches_path=matches_path)

    if not ret['success']:
        print(f"Localization FAILED for {query_image_name}")
        return []


    closest_image_id, relative_positions = calculate_sequential_positions(ret, reconstruction, num_images=50)
    closest_pose_name = reconstruction.images[closest_image_id].name
    print(f"Closest Pose ID: {closest_image_id}, Closest Pose Name: {closest_pose_name}")

    relative_positions_sorted = sorted(relative_positions, key=lambda x: get_image_number(x[0]))

    summary_data.append({'image_name': query_image_name, 'closest_pose_name': closest_pose_name, 'num_inliers': ret['num_inliers'], 'success': ret['success'], 'qvec': ret['qvec'].tolist(), 'tvec': ret['tvec'].tolist()})

    cprint.info(f'==================================RESULTS==================================')
    for i in relative_positions_sorted:
        print(i)

    csv_file_path = output_dir / f'{dataset_name}_summary.csv'
    csv_file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_file_path, 'w', newline='') as file:
        fieldnames = ['image_name', 'closest_pose_name', 'num_inliers', 'success', 'qvec', 'tvec']
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_data)

    fig = init_figure()

    viz_camera = Camera(model=query_camera.model_name, width=query_camera.width, height=query_camera.height, params=query_camera.params)
    pose = Image(tvec=ret['tvec'], qvec=ret['qvec'])
    plot_camera_colmap(fig, pose, viz_camera, name=query_image_name, color='green', fill=True)

    plot_reconstruction(fig, reconstruction, points_rgb=True)
    # plot_reconstruction(fig, reconstruction, ret['tvec'], points_rgb=True)

    fig.show()


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

    conf = {'estimation': {'ransac': {'max_error': 12}}, 'refinement': {'refine_focal_length': True, 'refine_extra_params': True}}
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

    process_query(query_image_path, localizer, camera_config, output_dir, reconstruction, dataset_name)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', help='Name of the dataset')
    parser.add_argument('--query_image_path', help='Path to the query image')
    args = parser.parse_args()

    query_processing(args.dataset, args.query_image_path)

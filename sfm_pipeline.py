import argparse
from pathlib import Path
from hloc import (
    extract_features, match_features, pairs_from_retrieval, pairs_from_ios_poses,
    reconstruction, localize_sfm, visualization
)
import h5py

# from hloc.utils.io import write_metadata

def run_sfm_pipeline(dataset_name, retrieval_conf, feature_conf, matcher_conf):
    dataset = Path('datasets') / dataset_name
    images = dataset / 'frames'
    outputs = Path('outputs') / dataset_name
    sfm_pairs = outputs / 'pairs-db-covis20.txt'
    loc_pairs = outputs / 'pairs-query-netvlad20.txt'
    reference_sfm = outputs / 'models' / 'sfm_superpoint+superglue'
    sfm_dir = outputs / 'models'

    sfm_dir.mkdir(parents=True, exist_ok=True)

    # Feature extraction
    print('=================Extract Features=================')
    feature_file = extract_features.main(feature_conf, images, outputs)
    feature_file_name = feature_conf['output']  # Use just the name of the feature file

    # Pairs from poses (optional, if you have pose information)
    print('=================Pairs From Poses=================')
    poses_file_path = dataset / 'poses.txt'
    if poses_file_path.exists():
        pairs_from_ios_poses.main(
            poses_file=poses_file_path,
            output=sfm_pairs,
            num_matched=1000000,
            rotation_threshold=10
        )

    # Before the matching step
    # features_path = '/home/ubuntu/Anantak/Pipelines/Hierarchical-Localization/outputs/testtrack/feats-disk.h5'
    # with h5py.File(features_path, 'r') as f:
    #     print(f"Keys in the features file: {list(f.keys())}")
        
    # with open(sfm_pairs, 'r') as file:
    #     for i, line in enumerate(file):
    #         if i > 10:  # Print only the first 10 lines
    #             break
    #         print(line.strip())

    # Feature matching
    print('=================Match Features=================')
    sfm_matches = match_features.main(matcher_conf, sfm_pairs, feature_file_name, outputs)  # Keep sfm_pairs as Path object
    matches_file_name = sfm_matches  # Use the string name of matches file

    # SFM reconstruction
    print('=================SFM Reconstruction=================')
    features_path = Path('/home/ubuntu/Anantak/Pipelines/Hierarchical-Localization/outputs/testtrack/feats-disk.h5')
    model = reconstruction.main(
        sfm_dir=sfm_dir,
        image_dir=images,
        pairs=sfm_pairs,
        features=features_path,
        matches=matches_file_name  # Use the string name
    )

    # Write reconstruction summary to info.txt
    # info_file = sfm_dir / 'info.txt'
    # write_metadata(model, info_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run SFM pipeline on a given dataset.')
    parser.add_argument('dataset_name', type=str, help='Name of the dataset folder')
    parser.add_argument('--retrieval_conf', type=str, default='netvlad', help='Configuration for image retrieval')
    parser.add_argument('--feature_conf', type=str, default='disk', help='Configuration for feature extraction')
    parser.add_argument('--matcher_conf', type=str, default='disk+lightglue', help='Configuration for feature matching')
    args = parser.parse_args()

    retrieval_conf = extract_features.confs[args.retrieval_conf]
    feature_conf = extract_features.confs[args.feature_conf]
    matcher_conf = match_features.confs[args.matcher_conf]

    run_sfm_pipeline(args.dataset_name, retrieval_conf, feature_conf, matcher_conf)

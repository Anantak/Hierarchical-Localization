import argparse
from pathlib import Path
from hloc import extract_features, pairs_from_retrieval, match_features, localize_sfm, visualization

def query_processing():
    # Paths for query processing
    base_dir = Path('datasets/office')
    output_dir = Path('outputs/office')
    db_img_dir = Path('datasets/office/frames')
    query_dir = Path('datasets/office/queries')
    reference_sfm = Path('datasets/office/frames/models/0')
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

    print("Query images processed and localized.")

if __name__ == '__main__':
    query_processing()

# Import necessary modules from hloc
from hloc import extract_features
from pathlib import Path

def main():
    query_dir = Path('datasets/office/queries')  # Directory containing query images
    output_dir = Path('outputs/office')          # Output directory for features
    retrieval_conf = extract_features.confs['netvlad']  # Configuration for feature extraction

    # Feature extraction for query images
    query_features_path = output_dir / 'feats-disk-queries.h5'  # Separate file for query features
    extract_features.main(retrieval_conf, query_dir, output_dir, feature_path=query_features_path)

if __name__ == '__main__':
    main()

import argparse
import numpy as np
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Union
from tqdm import tqdm
import pickle
import pycolmap

from . import logger
from .utils.io import get_keypoints, get_matches
from .utils.parsers import parse_image_lists, parse_retrieval


def do_covisibility_clustering(frame_ids: List[int],
                               reconstruction: pycolmap.Reconstruction):
    clusters = []
    visited = set()
    for frame_id in frame_ids:
        # Check if already labeled
        if frame_id in visited:
            continue

        # New component
        clusters.append([])
        queue = {frame_id}
        while len(queue):
            exploration_frame = queue.pop()

            # Already part of the component
            if exploration_frame in visited:
                continue
            visited.add(exploration_frame)
            clusters[-1].append(exploration_frame)

            observed = reconstruction.images[exploration_frame].points2D
            connected_frames = {
                obs.image_id
                for p2D in observed if p2D.has_point3D()
                for obs in
                reconstruction.points3D[p2D.point3D_id].track.elements
            }
            connected_frames &= set(frame_ids)
            connected_frames -= visited
            queue |= connected_frames

    clusters = sorted(clusters, key=len, reverse=True)
    return clusters


class QueryLocalizer:
    def __init__(self, reconstruction, config=None):
        self.reconstruction = reconstruction
        self.config = config or {}

    def localize(self, points2D_all, points2D_idxs, points3D_id, query_camera):
        points2D = points2D_all[points2D_idxs]
        points3D = [self.reconstruction.points3D[j].xyz for j in points3D_id]
        ret = pycolmap.absolute_pose_estimation(
            points2D, points3D, query_camera,
            estimation_options=self.config.get('estimation', {}),
            refinement_options=self.config.get('refinement', {}),
        )
        return ret


def pose_from_cluster(
        localizer: QueryLocalizer,
        qname: str,
        query_camera: pycolmap.Camera,
        db_ids: List[int],
        features_path: Path,
        matches_path: Path,
        **kwargs):

    # print(f"Processing query image: {qname}")
    kpq = get_keypoints(features_path, Path(qname).name)
    kpq += 0.5  # COLMAP coordinates

    kp_idx_to_3D = defaultdict(list)
    kp_idx_to_3D_to_db = defaultdict(lambda: defaultdict(list))
    num_matches = 0

    # print(f"Database IDs for {qname}: {db_ids}")

    for i, db_id in enumerate(db_ids):
        image = localizer.reconstruction.images[db_id]
        # print(f"Processing DB Image: {image.name}")
        # existing code...
        matches, _ = get_matches(matches_path, qname, image.name)
        if len(matches) == 0:
            print(f"No matches for query {qname} with DB image {image.name}")
            continue
        # print(f"{len(matches)} matches found for query {qname} with DB image {image.name}")

        points3D_ids = np.array([p.point3D_id if p.has_point3D() else -1
                                 for p in image.points2D])

        matches, _ = get_matches(matches_path, qname, image.name)

        if len(matches) == 0:
            # print(f"No matches found for query image {qname} with database image {image.name}")
            continue

        matches = matches[points3D_ids[matches[:, 1]] != -1]
        num_matches += len(matches)
        # if len(matches) > 0:
        #     print(f"Found {len(matches)} matches for query image {qname} with database image {image.name}")
        for idx, m in matches:
            id_3D = points3D_ids[m]
            kp_idx_to_3D_to_db[idx][id_3D].append(i)
            if id_3D not in kp_idx_to_3D[idx]:
                kp_idx_to_3D[idx].append(id_3D)

    print(f"Total matches found: {num_matches}")
    if num_matches == 0:
        print(f"No matches found for any database images with query image {qname}")
        return {'success': False}, {}

    idxs = list(kp_idx_to_3D.keys())
    mkp_idxs = [i for i in idxs for _ in kp_idx_to_3D[i]]
    mp3d_ids = [j for i in idxs for j in kp_idx_to_3D[i]]
    ret = localizer.localize(kpq, mkp_idxs, mp3d_ids, query_camera, **kwargs)

    # Logging and post-processing
    mkp_to_3D_to_db = [(j, kp_idx_to_3D_to_db[i][j])
                       for i in idxs for j in kp_idx_to_3D[i]]
    log = {
        'db': db_ids,
        'PnP_ret': ret,
        'keypoints_query': kpq[mkp_idxs],
        'points3D_ids': mp3d_ids,
        'points3D_xyz': None,
        'num_matches': num_matches,
        'keypoint_index_to_db': (mkp_idxs, mkp_to_3D_to_db),
    }

    # print('done')

    return ret, log



def main(reference_sfm: Union[Path, pycolmap.Reconstruction],
         queries: Path,
         retrieval: Path,
         features: Path,
         matches: Path,
         results: Path,
         ransac_thresh: int = 12,
         covisibility_clustering: bool = False,
         prepend_camera_name: bool = False,
         config: Dict = None):

    assert retrieval.exists(), retrieval
    assert features.exists(), features
    assert matches.exists(), matches

    queries = parse_image_lists(queries, with_intrinsics=True)
    retrieval_dict = parse_retrieval(retrieval)

    # print(retrieval_dict)
    # for i in retrieval_dict.keys():
    #     print(i)
    # quit()

    logger.info('Reading the 3D model...')
    if not isinstance(reference_sfm, pycolmap.Reconstruction):
        reference_sfm = pycolmap.Reconstruction(reference_sfm)
    db_name_to_id = {img.name: i for i, img in reference_sfm.images.items()}

    # print(db_name_to_id)


    config = {"estimation": {"ransac": {"max_error": ransac_thresh}},
              **(config or {})}
    localizer = QueryLocalizer(reference_sfm, config)

    poses = {}
    logs = {
        'features': features,
        'matches': matches,
        'retrieval': retrieval,
        'loc': {},
    }
    logger.info('Starting localization...')


    # print(retrieval_dict)
    # print(db_name_to_id)
    # quit()
    for qname, qcam in tqdm(queries):
        query_image_name = Path(qname).name
        if query_image_name not in retrieval_dict:
            logger.warning(f'No images retrieved for query image {query_image_name}. Skipping...')
            continue

        db_names = retrieval_dict[query_image_name]
        print(f"Query Image: {query_image_name}, DB Images: {db_names}")


        db_ids = []
        for db_name in db_names:
            if db_name in db_name_to_id:
                db_ids.append(db_name_to_id[db_name])
            else:
                logger.warning(f'Retrieved image {db_name} for query {query_image_name} not in database')
        print(f"DB IDs for {query_image_name}: {db_ids}")




        if covisibility_clustering:
            clusters = do_covisibility_clustering(db_ids, reference_sfm)
            best_inliers = 0
            best_cluster = None
            logs_clusters = []
            for i, cluster_ids in enumerate(clusters):
                ret, log = pose_from_cluster(
                        localizer, qname, qcam, cluster_ids, features, matches)
                if ret['success'] and ret['num_inliers'] > best_inliers:
                    best_cluster = i
                    best_inliers = ret['num_inliers']
                logs_clusters.append(log)
            if best_cluster is not None:
                ret = logs_clusters[best_cluster]['PnP_ret']
                poses[qname] = (ret['qvec'], ret['tvec'])
            logs['loc'][qname] = {
                'db': db_ids,
                'best_cluster': best_cluster,
                'log_clusters': logs_clusters,
                'covisibility_clustering': covisibility_clustering,
            }
        else:
            if db_ids:  # Check if db_ids list is not empty
                ret, log = pose_from_cluster(
                        localizer, qname, qcam, db_ids, features, matches)
                if ret['success']:
                    poses[qname] = (ret['qvec'], ret['tvec'])
                else:
                    closest = reference_sfm.images[db_ids[0]]
                    poses[qname] = (closest.qvec, closest.tvec)
            else:
                # Handle the case where db_ids is empty
                print(f"No database images found for query {qname}")
                continue  # Skip to the next iteration of the loop
            log['covisibility_clustering'] = covisibility_clustering


    logger.info(f'Localized {len(poses)} / {len(queries)} images.')
    logger.info(f'Writing poses to {results}...')
    with open(results, 'w') as f:
        for q in poses:
            qvec, tvec = poses[q]
            qvec = ' '.join(map(str, qvec))
            tvec = ' '.join(map(str, tvec))
            name = q.split('/')[-1]
            if prepend_camera_name:
                name = q.split('/')[-2] + '/' + name
            f.write(f'{name} {qvec} {tvec}\n')

    logs_path = f'{results}_logs.pkl'
    logger.info(f'Writing logs to {logs_path}...')
    with open(logs_path, 'wb') as f:
        pickle.dump(logs, f)
    logger.info('Done!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--reference_sfm', type=Path, required=True)
    parser.add_argument('--queries', type=Path, required=True)
    parser.add_argument('--features', type=Path, required=True)
    parser.add_argument('--matches', type=Path, required=True)
    parser.add_argument('--retrieval', type=Path, required=True)
    parser.add_argument('--results', type=Path, required=True)
    parser.add_argument('--ransac_thresh', type=float, default=12.0)
    parser.add_argument('--covisibility_clustering', action='store_true')
    parser.add_argument('--prepend_camera_name', action='store_true')
    args = parser.parse_args()
    main(**args.__dict__)

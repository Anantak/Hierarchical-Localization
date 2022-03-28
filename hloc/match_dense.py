from tqdm import tqdm
import numpy as np
import h5py
import torch
from pathlib import Path
from typing import Dict, Optional, List
import pprint
import argparse
import torchvision.transforms.functional as F
from omegaconf import OmegaConf
from collections import defaultdict, Iterable

from . import matchers, logger
from .utils.base_model import dynamic_load
from .utils.parsers import parse_retrieval, names_to_pair
from .match_features import find_unique_new_pairs
from .extract_features import read_image, resize_image
from .utils.io import list_h5_names

confs = {
    'loftr': {
        'output': 'matches-loftr',
        'model': {
            'name': 'loftr',
            'weights': 'outdoor'
        },
        'preprocessing': {
            'grayscale': True,
            'resize_max': 1024,
            'dfactor': 8
        },
        'psize': 1,
    },
}


def get_grouped_ids(array):
    # Group array indices based on its values
    # all duplicates are grouped as a set
    idx_sort = np.argsort(array)
    sorted_array = array[idx_sort]
    _, ids, _ = np.unique(sorted_array, return_counts=True,
                          return_index=True)
    res = np.split(idx_sort, ids[1:])
    return res


def get_unique_matches(match_ids, scores):
    if len(match_ids.shape) == 1:
        return [0]

    isets1 = get_grouped_ids(match_ids[:, 0])
    isets2 = get_grouped_ids(match_ids[:, 0])
    uid1s = [ids[scores[ids].argmax()] for ids in isets1 if len(ids) > 0]
    uid2s = [ids[scores[ids].argmax()] for ids in isets2 if len(ids) > 0]
    uids = list(set(uid1s).intersection(uid2s))
    return match_ids[uids], scores[uids]


def to_cpts(kpts, ps):
    cpts = np.round(np.round((kpts + 0.5) / ps) * ps - 0.5, 2)
    return [tuple(cpt) for cpt in cpts]


def quantize_keypoints(kpts, other_cpts, ps, extend=True):
    kpt_ids = []
    cpts = to_cpts(kpts, ps)
    cp_to_id = {val: i for i, val in enumerate(other_cpts)}
    for i, cpt in enumerate(cpts):
        try:
            kid = cp_to_id[cpt]
        except KeyError:
            if extend:
                kid = len(cp_to_id)
                cp_to_id[cpt] = kid
                other_cpts.append(cpt)
            else:
                kid = -1
        kpt_ids.append(kid)
    return np.array(kpt_ids)


def matches_to_matches0(matches, scores):
    if matches.shape[0] == 0:
        return (np.zeros([0, 2], dtype=np.uint32),
                np.zeros([0], dtype=np.float32))
    n_kps0 = np.max(matches[:, 0]) + 1
    matches0 = -np.ones((n_kps0,))
    scores0 = np.zeros((n_kps0,))
    matches0[matches[:, 0]] = matches[:, 1]
    scores0[matches[:, 0]] = scores
    return matches0.astype(np.int32), scores0.astype(np.float16)


def scale_keypoints(kpts, scale):
    if np.any(scale != 1.0):
        kpts *= kpts.new_tensor(scale)
    return kpts


class ImagePairDataset(torch.utils.data.Dataset):
    default_conf = {
        'grayscale': True,
        'resize_max': 1024,
        'dfactor': 8,
        'cache_images': False,
    }

    def __init__(self, image_dir, conf, pairs):
        self.image_dir = image_dir
        self.conf = OmegaConf.merge(self.default_conf, conf)
        self.pairs = pairs
        if self.conf.cache_images:
            image_names = set(sum(pairs, ()))  # unique image names in pairs
            logger.info(
                f'Loading and caching {len(image_names)} unique images.')
            self.images = {}
            self.scales = {}
            for name in tqdm(image_names):
                image = read_image(self.image_dir / name, self.conf.grayscale)
                self.images[name], self.scales[name] = self.preprocess(image)

    def preprocess(self, image: np.ndarray):
        image = image.astype(np.float32, copy=False)
        size = image.shape[:2][::-1]
        scale = np.array([1.0, 1.0])

        if self.conf.resize_max:
            scale = self.conf.resize_max / max(size)
            if scale < 1.0:
                size_new = tuple(int(round(x*scale)) for x in size)
                image = resize_image(image, size_new, 'cv2_area')
                scale = np.array(size) / np.array(size_new)

        if self.conf.grayscale:
            assert image.ndim == 2, image.shape
            image = image[None]
        else:
            image = image.transpose((2, 0, 1))  # HxWxC to CxHxW
        image = torch.from_numpy(image / 255.0).float()

        # assure that the size is divisible by dfactor
        size_new = tuple(map(
                lambda x: int(x // self.conf.dfactor * self.conf.dfactor),
                image.shape[-2:]))
        image = F.resize(image, size=size_new)
        scale = np.array(size) / np.array(size_new)[::-1]
        return image, scale

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        name0, name1 = self.pairs[idx]
        if self.conf.cache_images:
            image0, scale0 = self.images[name0], self.scales[name0]
            image1, scale1 = self.images[name1], self.scales[name1]
        else:
            image0 = read_image(self.image_dir / name0, self.conf.grayscale)
            image1 = read_image(self.image_dir / name1, self.conf.grayscale)
            image0, scale0 = self.preprocess(image0)
            image1, scale1 = self.preprocess(image1)
        return image0, image1, scale0, scale1, name0, name1


@torch.no_grad()
def match_dense_from_paths(conf: Dict,
                           pairs_path: Path,
                           image_dir: Path,
                           match_path: Path,  # out
                           feature_path_q: Path,
                           feature_paths_refs: Optional[List[Path]] = [],
                           overwrite: bool = False) -> Path:
    conf = OmegaConf.merge({'psize': 1}, conf)
    for path in feature_paths_refs:
        if not path.exists():
            raise FileNotFoundError(f'Reference feature file {path}.')
    pairs = parse_retrieval(pairs_path)
    pairs = [(q, r) for q, rs in pairs.items() for r in rs]
    pairs = find_unique_new_pairs(pairs, None if overwrite else match_path)
    all_names = set(sum(pairs, ()))

    name2ref = {n: i for i, p in enumerate(feature_paths_refs)
                for n in list_h5_names(p)}
    existing_refs = all_names.intersection(set(name2ref.keys()))
    required_queries = all_names - existing_refs

    if feature_path_q.exists():
        existing_names_q = set(list_h5_names(feature_path_q))
        required_queries = required_queries - existing_names_q

    if len(pairs) == 0 and len(required_queries) == 0:
        logger.info("All pairs exist. Skipping dense matching.")
        return

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    Model = dynamic_load(matchers, conf['model']['name'])
    model = Model(conf['model']).eval().to(device)

    # Load query keypoins
    kpdict = defaultdict(list)
    for name in existing_refs:
        with h5py.File(str(feature_paths_refs[name2ref[name]]), 'r') as fd:
            kpdict[name] = to_cpts(fd[name]['keypoints'].__array__(), conf.psize)

    dataset = ImagePairDataset(image_dir, conf["preprocessing"], pairs)
    loader = torch.utils.data.DataLoader(
            dataset, num_workers=1, batch_size=1, shuffle=False)
    logger.info("Performing dense matching...")
    with h5py.File(str(match_path), 'a') as fd:
        for data in tqdm(loader):
            # match semi-dense
            image0, image1, scale0, scale1, (name0,), (name1,) = data
            scale0, scale1 = scale0[0].numpy(), scale1[0].numpy()
            data = {'image0': image0.to(device), 'image1': image1.to(device)}
            pred = model(data)
            kpts0, kpts1 = pred["keypoints0"], pred["keypoints1"]
            kpts0 = scale_keypoints(kpts0 + 0.5, scale0) - 0.5
            kpts1 = scale_keypoints(kpts1 + 0.5, scale1) - 0.5
            kpts0 = kpts0.cpu().numpy()
            kpts1 = kpts1.cpu().numpy()

            # Aggregate local features
            q0 = name0 in required_queries
            q1 = name1 in required_queries
            kpt_ids0 = quantize_keypoints(kpts0, kpdict[name0], conf.psize, q0)
            kpt_ids1 = quantize_keypoints(kpts1, kpdict[name1], conf.psize, q1)
            valid = (kpt_ids0 != -1) & (kpt_ids1 != -1)
            matches = np.dstack([kpt_ids0[valid], kpt_ids1[valid]])
            matches = matches.reshape(-1, 2)
            scores = pred['scores'].cpu().numpy()[valid]
            # Remove n-to-1 matches
            matches, scores = get_unique_matches(matches, scores)
            matches0, scores0 = matches_to_matches0(matches, scores)
            pair = names_to_pair(name0, name1)
            if pair in fd:
                del fd[pair]
            grp = fd.create_group(pair)

            grp.create_dataset('matches0', data=matches0)
            grp.create_dataset('matching_scores0', data=scores0)
    if len(required_queries) > 0:
        # Save keypoints
        with h5py.File(feature_path_q, 'a') as fd:
            logger.info(f'Save keypoints of {len(required_queries)} images...')
            n_kps = 0
            for name in tqdm(required_queries, smoothing=.1):
                kps = np.array(kpdict[name], dtype=np.float32)
                kgrp = fd.create_group(name)
                kgrp.create_dataset('keypoints', data=kps)
                n_kps += kps.shape[0]
        avg_kp_per_image = round(n_kps / len(required_queries), 1)
        logger.info(f'Finished assignment, found {avg_kp_per_image} '
                    f'keypoints/image (avg.), total {n_kps}.')


@torch.no_grad()
def main(conf: Dict,
         pairs: Path,
         image_dir: Path,
         export_dir: Optional[Path] = None,
         matches: Optional[Path] = None,  # out
         features: Optional[Path] = None,  # out
         features_ref: Optional[Path] = None,
         overwrite: bool = False) -> Path:
    logger.info('Extracting local features with configuration:'
                f'\n{pprint.pformat(conf)}')

    if features is None:
        features = 'feats_'

    if isinstance(features, Path):
        features_q = features
        if matches is None:
            raise ValueError('Either provide both features and matches as Path'
                             ' or both as names.')
    else:
        if export_dir is None:
            raise ValueError('Provide an export_dir if features and matches'
                             f' are not file paths: {features}, {matches}.')
        features_q = Path(export_dir, f'{features}_{conf["output"]}_.h5')
        if matches is None:
            matches = Path(
                export_dir, f'{conf["output"]}_{pairs.stem}.h5')

    if features_ref is None:
        features_ref = []
    if isinstance(features_ref, Iterable):
        features_ref = list(features_ref)
    elif isinstance(features_ref, Path):
        features_ref = [features_ref]
    else:
        raise TypeError(str(features_ref))

    match_dense_from_paths(conf, pairs, image_dir, matches,
                           features_q, features_ref, overwrite)

    return features_q, matches


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pairs', type=Path, required=True)
    parser.add_argument('--image_dir', type=Path, required=True)
    parser.add_argument('--export_dir', type=Path, required=True)
    parser.add_argument('--matches', type=Path,
                        default=confs['loftr']['output'])
    parser.add_argument('--features', type=str,
                        default='feats_' + confs['loftr']['output'])
    parser.add_argument('--conf', type=str, default='loftr',
                        choices=list(confs.keys()))
    args = parser.parse_args()
    main(confs[args.conf], args.pairs, args.image_dir, args.export_dir,
         args.matches, args.features)
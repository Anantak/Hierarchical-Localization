import argparse
from typing import Union, Optional, Dict, List, Tuple
from pathlib import Path
import pprint
from queue import Queue
from threading import Thread
from functools import partial
from tqdm import tqdm
import h5py
import torch

from . import matchers, logger
from .utils.base_model import dynamic_load
from .utils.parsers import names_to_pair, names_to_pair_old, parse_retrieval


'''
A set of standard configurations that can be directly selected from the command
line using their name. Each is a dictionary with the following entries:
    - output: the name of the match file that will be generated.
    - model: the model configuration, as passed to a feature matcher.
'''
confs = {
    'superpoint+lightglue': {
        'output': 'matches-superpoint-lightglue',
        'model': {
            'name': 'lightglue',
            'features': 'superpoint',
        },
    },
    'disk+lightglue': {
        'output': 'matches-disk-lightglue',
        'model': {
            'name': 'lightglue',
            'features': 'disk',
        },
    },
    'superglue': {
        'output': 'matches-superglue',
        'model': {
            'name': 'superglue',
            'weights': 'outdoor',
            'sinkhorn_iterations': 50,
        },
    },
    'superglue-fast': {
        'output': 'matches-superglue-it5',
        'model': {
            'name': 'superglue',
            'weights': 'outdoor',
            'sinkhorn_iterations': 5,
        },
    },
    'NN-superpoint': {
        'output': 'matches-NN-mutual-dist.7',
        'model': {
            'name': 'nearest_neighbor',
            'do_mutual_check': True,
            'distance_threshold': 0.7,
        },
    },
    'NN-ratio': {
        'output': 'matches-NN-mutual-ratio.8',
        'model': {
            'name': 'nearest_neighbor',
            'do_mutual_check': True,
            'ratio_threshold': 0.8,
        }
    },
    'NN-mutual': {
        'output': 'matches-NN-mutual',
        'model': {
            'name': 'nearest_neighbor',
            'do_mutual_check': True,
        },
    },
    'adalam': {
        'output': 'matches-adalam',
        'model': {
            'name': 'adalam'
        },
    }
}


class WorkQueue():
    def __init__(self, work_fn, num_threads=1):
        self.queue = Queue(num_threads)
        self.threads = [
            Thread(target=self.thread_fn, args=(work_fn,))
            for _ in range(num_threads)
        ]
        for thread in self.threads:
            thread.start()

    def join(self):
        for thread in self.threads:
            self.queue.put(None)
        for thread in self.threads:
            thread.join()

    def thread_fn(self, work_fn):
        item = self.queue.get()
        while item is not None:
            work_fn(item)
            item = self.queue.get()

    def put(self, data):
        self.queue.put(data)


class FeaturePairsDataset(torch.utils.data.Dataset):
    def __init__(self, pairs, feature_path_dataset, feature_path_queries):
        self.pairs = pairs
        self.feature_path_dataset = feature_path_dataset
        self.feature_path_queries = feature_path_queries

    def __getitem__(self, idx):
        name0, name1 = self.pairs[idx]

        # # Append '.jpg' extension if not present
        # name0_with_ext = name0 if name0.endswith('.jpg') else name0 + '.jpg'
        name0_with_ext = name0 if name0.endswith('.jpeg') else name0 + '.jpg'
        name1_with_ext = name1 if name1.endswith('.jpg') else name1 + '.jpg'

        data = {}

        # Process first image
        feature_path1 = self.feature_path_dataset
        feature_path0= self.feature_path_queries

        # print(feature_path0)
        # print(feature_path1)
        # print(f" Accessing {name0_with_ext} from {feature_path0}")  # Added print statement


        with h5py.File(feature_path0, 'r') as fd:
            grp = fd[name0_with_ext]  # Use the modified name
            data['keypoints0'] = torch.from_numpy(grp['keypoints'].__array__()).float()
            data['descriptors0'] = torch.from_numpy(grp['descriptors'].__array__()).float()
            data['image0'] = torch.empty((1,)+tuple(grp['image_size'])[::-1])

        # Process second image
        # feature_path1 = self.feature_path_queries if 'test' in name1 else self.feature_path_dataset
            
        # print(f" Accessing {name1_with_ext} from {feature_path1}")  # Added print statement
        with h5py.File(feature_path1, 'r') as fd:
            grp = fd[name1_with_ext]  # Use the modified name
            data['keypoints1'] = torch.from_numpy(grp['keypoints'].__array__()).float()
            data['descriptors1'] = torch.from_numpy(grp['descriptors'].__array__()).float()
            data['image1'] = torch.empty((1,)+tuple(grp['image_size'])[::-1])

        return data

    def __len__(self):
        return len(self.pairs)



def writer_fn(inp, match_path):
    pair, pred = inp
    num_matches = (pred['matches0'][0] > -1).sum().item()
    # print(f"Writing {num_matches} matches for pair {pair} to file")
    # Rest of the code remains the same...

    pair, pred = inp
    with h5py.File(str(match_path), 'a', libver='latest') as fd:
        if pair in fd:
            del fd[pair]
        grp = fd.create_group(pair)
        matches = pred['matches0'][0].cpu().short().numpy()
        grp.create_dataset('matches0', data=matches)
        if 'matching_scores0' in pred:
            scores = pred['matching_scores0'][0].cpu().half().numpy()
            grp.create_dataset('matching_scores0', data=scores)


def main(conf: Dict,
         pairs: Path, features: Union[Path, str],
         export_dir: Optional[Path] = None,
         matches: Optional[Path] = None,
         features_ref: Optional[Path] = None,
         overwrite: bool = False) -> Path:

    if isinstance(features, Path) or Path(features).exists():
        features_q = features
        if matches is None:
            raise ValueError('Either provide both features and matches as Path'
                             ' or both as names.')
    else:
        if export_dir is None:
            raise ValueError('Provide an export_dir if features is not'
                             f' a file path: {features}.')
        features_q = Path(export_dir, features+'.h5')
        if matches is None:
            matches = Path(
                export_dir, f'{features}_{conf["output"]}_{pairs.stem}.h5')

    if features_ref is None:
        features_ref = features_q
    match_from_paths(conf, pairs, matches, features_q, features_ref, overwrite)

    return matches


def find_unique_new_pairs(pairs_all: List[Tuple[str]], match_path: Path = None):
    '''Avoid to recompute duplicates to save time.'''
    pairs = set()
    for i, j in pairs_all:
        if (j, i) not in pairs:
            pairs.add((i, j))
    pairs = list(pairs)
    if match_path is not None and match_path.exists():
        with h5py.File(str(match_path), 'r', libver='latest') as fd:
            pairs_filtered = []
            for i, j in pairs:
                if (names_to_pair(i, j) in fd or
                        names_to_pair(j, i) in fd or
                        names_to_pair_old(i, j) in fd or
                        names_to_pair_old(j, i) in fd):
                    continue
                pairs_filtered.append((i, j))
        return pairs_filtered
    return pairs


@torch.no_grad()
def match_from_paths(conf: Dict,
                     pairs_path: Path,
                     match_path: Path,
                     feature_path_q: Path,
                     feature_path_ref: Path,
                     overwrite: bool = False) -> Path:
    logger.info('Matching local features with configuration:'
                f'\n{pprint.pformat(conf)}')

    if not feature_path_q.exists():
        raise FileNotFoundError(f'Query feature file {feature_path_q}.')
    if not feature_path_ref.exists():
        raise FileNotFoundError(f'Reference feature file {feature_path_ref}.')
    match_path.parent.mkdir(exist_ok=True, parents=True)

    assert pairs_path.exists(), pairs_path
    pairs = parse_retrieval(pairs_path)
    pairs = [(q, r) for q, rs in pairs.items() for r in rs]
    pairs = find_unique_new_pairs(pairs, None if overwrite else match_path)
    if len(pairs) == 0:
        logger.info('Skipping the matching.')
        return

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    Model = dynamic_load(matchers, conf['model']['name'])
    model = Model(conf['model']).eval().to(device)

    # Example usage
    dataset = FeaturePairsDataset(pairs, feature_path_dataset='/home/ubuntu/Anantak/Pipelines/Hierarchical-Localization/outputs/testtrack/feats-disk.h5', feature_path_queries='/home/ubuntu/Anantak/Pipelines/Hierarchical-Localization/outputs/testtrack/feats-disk-queries.h5')

    loader = torch.utils.data.DataLoader(
        dataset, num_workers=5, batch_size=1, shuffle=False, pin_memory=True)
    writer_queue = WorkQueue(partial(writer_fn, match_path=match_path), 5)

    for idx, data in enumerate(tqdm(loader, smoothing=.1)):
        data = {k: v if k.startswith('image')
                else v.to(device, non_blocking=True) for k, v in data.items()}
        pred = model(data)
        pair = names_to_pair(*pairs[idx])
        writer_queue.put((pair, pred))

        num_matches = (pred['matches0'][0] > -1).sum().item()
        # print(f"Matching {pairs[idx][0]} with {pairs[idx][1]}: {num_matches} matches found")

    writer_queue.join()

    with h5py.File(str(match_path), 'r') as fd:
        for group in fd:
            for subgroup in fd[group]:
                matches = fd[group][subgroup]['matches0'][:]
                num_matches = (matches > -1).sum()
                # if num_matches > 0:
                    # print(f"Found {num_matches} matches in pair {group} - {subgroup}")

    logger.info('Finished exporting matches.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pairs', type=Path, required=True)
    parser.add_argument('--export_dir', type=Path)
    parser.add_argument('--features', type=str,
                        default='feats-superpoint-n4096-r1024')
    parser.add_argument('--matches', type=Path)
    parser.add_argument('--conf', type=str, default='superglue',
                        choices=list(confs.keys()))
    args = parser.parse_args()
    main(confs[args.conf], args.pairs, args.features, args.export_dir)

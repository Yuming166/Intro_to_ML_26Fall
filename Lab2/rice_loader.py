"""Fixed 60/20/20 stratified image-feature split; train-only standardization."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import hashlib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

FEATURES = ['Area', 'Perimeter', 'Major_Axis_Length', 'Minor_Axis_Length',
            'Eccentricity', 'Convex_Area', 'Extent']
CLASS_NAMES = ['Cammeo', 'Osmancik']


@dataclass
class Split:
    x: np.ndarray
    y: np.ndarray
    original_rows: np.ndarray


def resolve_data_path(data_dir=None):
    if data_dir is not None:
        p = Path(data_dir).expanduser().resolve()
        return p if p.suffix.lower() == '.csv' else p / 'rice.csv'
    script_dir = Path(__file__).resolve().parent
    for folder in [script_dir / 'data', script_dir.parent / 'data', script_dir]:
        if (folder / 'rice.csv').is_file():
            return folder / 'rice.csv'
    raise FileNotFoundError('请将 rice.csv 放入脚本旁的 data 文件夹，或指定 --data-dir')


def load_data(data_dir=None, scaling='standard', split_seed=42):
    path = resolve_data_path(data_dir)
    frame = pd.read_csv(path)
    if list(frame.columns) != FEATURES + ['Class'] or len(frame) != 3810:
        raise ValueError('Expected the official Rice dataset: 3810 rows, 7 features and Class')
    if frame.isna().any().any() or set(frame['Class']) != set(CLASS_NAMES):
        raise ValueError('Missing values or unexpected labels')
    x = frame[FEATURES].to_numpy(dtype=np.float64)
    y = frame['Class'].map({'Cammeo': -1, 'Osmancik': 1}).to_numpy(dtype=np.int64)
    if not np.isfinite(x).all() or len(np.unique(x, axis=0)) != len(x):
        raise ValueError('Expected finite, nonduplicated feature rows')
    ids = np.arange(len(y))
    development, test_ids = train_test_split(ids, test_size=0.2, random_state=split_seed, stratify=y)
    train_ids, val_ids = train_test_split(development, test_size=0.25, random_state=split_seed + 1,
                                         stratify=y[development])
    mean = x[train_ids].mean(axis=0)
    std = x[train_ids].std(axis=0)
    std = np.where(std == 0.0, 1.0, std)
    if scaling == 'standard':
        transformed = (x - mean) / std
    elif scaling == 'none':
        transformed = x.copy()
    else:
        raise ValueError('scaling must be standard or none')
    splits = {name: Split(transformed[ix], y[ix], ix) for name, ix in
              [('train', train_ids), ('validation', val_ids), ('test', test_ids)]}
    sets = [set(s.original_rows) for s in splits.values()]
    assert len(sets[0] & sets[1]) == len(sets[0] & sets[2]) == len(sets[1] & sets[2]) == 0
    assert set.union(*sets) == set(ids)
    meta = {'source': 'https://archive.ics.uci.edu/dataset/545/rice+cammeo+and+osmancik',
            'data_file': str(path), 'data_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
            'features': FEATURES, 'label_map': {'Cammeo': -1, 'Osmancik': 1},
            'split_seed': split_seed, 'split_fractions': [0.6, 0.2, 0.2],
            'split_counts': {k: len(s.y) for k, s in splits.items()},
            'class_counts': {k: {name: int(np.sum(s.y == code)) for name, code in
                                   [('Cammeo', -1), ('Osmancik', 1)]} for k, s in splits.items()},
            'scaling': scaling, 'standardizer_fit_on': 'training rows only',
            'training_mean': mean.tolist(), 'training_std_ddof0': std.tolist(),
            'row_overlap': 0, 'exact_feature_duplicates': 0,
            'scope': 'Row-level stratified holdout; no farm, acquisition-batch or variety-transfer independence claimed'}
    return splits, meta

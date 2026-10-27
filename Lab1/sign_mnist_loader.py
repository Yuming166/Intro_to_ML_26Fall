"""已提供完整实现，无补写题。保留官方测试集，仅从官方训练集划分训练／验证。"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


@dataclass
class Split:
    x: np.ndarray
    y: np.ndarray
    original_rows: np.ndarray


def resolve_data_dir(data_dir=None):
    """显式路径优先；默认先找脚本旁的 data/，兼容 CSV 与脚本同层。"""
    if data_dir is not None:
        return Path(data_dir).expanduser().resolve()
    script_dir = Path(__file__).resolve().parent
    names = ('sign_mnist_train.csv', 'sign_mnist_test.csv')
    for candidate in (script_dir / 'data', script_dir):
        if all((candidate / name).is_file() for name in names):
            return candidate
    return script_dir / 'data'


def load_data(data_dir=None, validation_fraction=0.1, seed=42, normalization='unit'):
    data_dir = resolve_data_dir(data_dir)
    if not 0 < validation_fraction < 1:
        raise ValueError('validation_fraction must be between 0 and 1')
    paths = {name: data_dir / f'sign_mnist_{name}.csv' for name in ('train', 'test')}
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(f'{path}: 请将两个官方 CSV 放在此目录，或通过 --data-dir 指定其所在文件夹')
    hashes = {name: sha256(path) for name, path in paths.items()}
    cache = data_dir / 'arrays_cache.npz'
    source_hashes = None
    if cache.exists():
        with np.load(cache, allow_pickle=False) as archive:
            source_hashes = json.loads(str(archive['source_hashes']))
            if source_hashes == hashes:
                raw = {name: archive[name].copy() for name in ('x_train', 'y_train', 'x_test', 'y_test')}
    if source_hashes != hashes:
        frames = {name: pd.read_csv(path, dtype=np.uint8) for name, path in paths.items()}
        expected_columns = ['label'] + [f'pixel{i}' for i in range(1, 785)]
        for name, frame in frames.items():
            if list(frame.columns) != expected_columns:
                raise ValueError(f'{name}: unexpected CSV schema')
            if frame.isna().any().any():
                raise ValueError(f'{name}: missing values')
        raw = {f'{kind}_{name}': (frame.iloc[:, 1:].to_numpy(copy=True) if kind == 'x'
                                  else frame['label'].to_numpy(copy=True))
               for name, frame in frames.items() for kind in ('x', 'y')}
        np.savez_compressed(cache, source_hashes=np.array(json.dumps(hashes)), **raw)
    if len(raw['y_train']) != 27455 or len(raw['y_test']) != 7172:
        raise ValueError('expected the official 27455 / 7172 train/test split')
    labels = np.unique(raw['y_train'])
    expected_labels = np.array([i for i in range(26) if i not in (9, 25)], dtype=np.uint8)
    if not np.array_equal(labels, expected_labels):
        raise ValueError('expected 24 ASL letters excluding J and Z')
    if not set(raw['y_test']).issubset(set(labels)):
        raise ValueError('unseen test class')
    # The raw labels are NOT contiguous. Fit the mapping on the training labels.
    y_train = np.searchsorted(labels, raw['y_train']).astype(np.int64)
    y_test = np.searchsorted(labels, raw['y_test']).astype(np.int64)
    indices = np.arange(len(y_train))
    train_indices, val_indices = train_test_split(indices, test_size=validation_fraction,
                                                   random_state=seed, stratify=y_train)
    if normalization not in ('unit', 'center', 'image-center'):
        raise ValueError('normalization must be unit, center or image-center')
    x_train = raw['x_train'].astype(np.float32) / np.float32(255)
    x_test = raw['x_test'].astype(np.float32) / np.float32(255)
    if normalization == 'center':
        x_train -= np.float32(.5)
        x_test -= np.float32(.5)
    elif normalization == 'image-center':
        x_train -= x_train.mean(axis=1, keepdims=True)
        x_test -= x_test.mean(axis=1, keepdims=True)
    splits = {'train': Split(x_train[train_indices], y_train[train_indices], train_indices),
              'validation': Split(x_train[val_indices], y_train[val_indices], val_indices),
              'test': Split(x_test, y_test, np.arange(len(y_test)))}
    assert not set(train_indices) & set(val_indices)
    assert set(train_indices) | set(val_indices) == set(indices)
    letters = [chr(ord('A') + int(label)) for label in labels]
    metadata = {'source': 'Kaggle datamunge/sign-language-mnist',
                'dataset_url': 'https://www.kaggle.com/datasets/datamunge/sign-language-mnist',
                'csv_sha256': hashes, 'official_train_rows': len(y_train), 'official_test_rows': len(y_test),
                'validation_drawn_from_official_train_only': True, 'validation_fraction': validation_fraction,
                'split_seed': seed, 'normalization': normalization,
                'normalization_formula': {'unit': 'float32(pixel)/255', 'center': 'float32(pixel)/255-.5',
                                          'image-center': 'pixel/255 - mean(pixel/255) within each image'}[normalization],
                'normalization_fitted_on_test': False,
                'raw_label_to_class': {str(int(label)): index for index, label in enumerate(labels)},
                'class_to_letter': letters, 'input_dimension': 784, 'classes': 24,
                'split_counts': {name: len(split.y) for name, split in splits.items()},
                'class_counts': {name: np.bincount(split.y, minlength=24).tolist() for name, split in splits.items()},
                'validation_limitation': 'Random stratified image split; augmented-image family and signer identity groups unavailable. Not signer-disjoint validation.'}
    return splits, metadata

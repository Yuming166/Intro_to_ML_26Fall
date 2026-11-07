"""实验框架：读取随包CSV，固定官方划分；默认不打开测试集文件。"""

from collections import Counter, defaultdict
import csv
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re

import numpy as np


@dataclass
class Split:
    texts: list[str]
    y: np.ndarray


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def normalized(text):
    return re.sub(r"\s+", " ", text.strip().lower())


def _read_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != ["text", "label"]:
            raise ValueError(f"{path.name} 的列必须为 text,label")
        rows = [(row["text"], row["label"]) for row in reader]
    if any(not text.strip() or not label.strip() for text, label in rows):
        raise ValueError(f"{path.name} 含空文本或空标签")
    return rows


def load_data(data_dir=None, include_test=False):
    folder = Path(data_dir).expanduser().resolve() if data_dir else Path(__file__).resolve().parent / "data"
    metadata = json.loads((folder / "SOURCE.json").read_text(encoding="utf-8"))
    names = ["train", "val"] + (["test"] if include_test else [])
    raw = {}
    for name in names:
        path = folder / f"{name}.csv"
        if sha256(path) != metadata["csv_sha256"][path.name]:
            raise ValueError(f"{path.name} 与固定数据快照不一致，请恢复随包数据")
        raw[name] = _read_csv(path)
    class_names = sorted({label for text, label in raw["train"]})
    if len(class_names) != 150 or "oos" in class_names:
        raise ValueError("本实验训练集应恰好包含150个已知意图")
    mapping = {name: i for i, name in enumerate(class_names)}
    expected = {"train": 100, "val": 20, "test": 30}
    splits, overlaps = {}, {}
    train_lookup = defaultdict(list)
    for index, (text, label) in enumerate(raw["train"]):
        train_lookup[normalized(text)].append({"index": index, "label": label})
    for name, rows in raw.items():
        counts = Counter(label for text, label in rows)
        if set(counts) != set(class_names) or set(counts.values()) != {expected[name]}:
            raise ValueError(f"{name} 不符合官方完整150类划分")
        splits[name] = Split([text for text, label in rows], np.array([mapping[label] for text, label in rows], dtype=np.int64))
        if name != "train":
            overlaps[name] = [{"row_index": i, "text": text, "label": label, "train_rows": train_lookup[normalized(text)]} for i, (text, label) in enumerate(rows) if normalized(text) in train_lookup]
    audit = {"official_rows_preserved": True, "classes": 150, "loaded_splits": {name: len(split.y) for name, split in splits.items()}, "test_file_opened": include_test, "oos_used": False, "train_text_overlaps": overlaps, "csv_sha256": {f"{name}.csv": metadata["csv_sha256"][f"{name}.csv"] for name in names}}
    return splits, class_names, audit, folder


def no_train_overlap_mask(train_texts, evaluation_texts):
    train = {normalized(text) for text in train_texts}
    return np.array([normalized(text) not in train for text in evaluation_texts], dtype=bool)

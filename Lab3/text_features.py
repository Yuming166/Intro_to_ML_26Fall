"""文本特征框架：训练集建词表，其他集合只能 transform。"""

import json
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer


TOKEN_PATTERN = r"(?u)\b\w+(?:'\w+)?\b"


class TextFeatures:
    def __init__(self, ngram_max=1, stopwords=None, vocabulary=None):
        if ngram_max not in (1, 2):
            raise ValueError("ngram_max 只能为 1 或 2")
        self.ngram_max = int(ngram_max)
        self.stopwords = None if stopwords is None else sorted(set(stopwords))
        self.vectorizer = CountVectorizer(
            lowercase=True,
            token_pattern=TOKEN_PATTERN,
            ngram_range=(1, self.ngram_max),
            stop_words=self.stopwords,
            vocabulary=vocabulary,
            dtype=np.float64,
        )

    def fit_transform(self, train_texts):
        return self.vectorizer.fit_transform(train_texts).tocsr()

    def transform(self, texts):
        return self.vectorizer.transform(texts).tocsr()

    def names(self):
        return self.vectorizer.get_feature_names_out()

    def save(self, path):
        payload = {
            "ngram_max": self.ngram_max,
            "stopwords": self.stopwords,
            "token_pattern": TOKEN_PATTERN,
            "vocabulary": {k: int(v) for k, v in self.vectorizer.vocabulary_.items()},
        }
        Path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path):
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if payload["token_pattern"] != TOKEN_PATTERN:
            raise ValueError("词元规则与保存模型不一致")
        return cls(payload["ngram_max"], payload["stopwords"], payload["vocabulary"])


def matrix_info(x):
    size = int(x.shape[0] * x.shape[1])
    return {
        "shape": list(x.shape),
        "nonzero_entries": int(x.nnz),
        "density": float(x.nnz / size) if size else 0.0,
        "csr_bytes": int(x.data.nbytes + x.indices.nbytes + x.indptr.nbytes),
        "empty_documents": int(np.count_nonzero(np.diff(x.indptr) == 0)),
    }

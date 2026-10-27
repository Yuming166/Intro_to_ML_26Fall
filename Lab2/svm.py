"""NumPy linear hinge/logistic classifiers; mini-batch stochastic (sub)gradients.

Objective: regularization/2 * ||w||^2 + mean(loss(y * (X @ w + b))).
The intercept is not regularized. X=[m,d], w=[d], y=[m], b is scalar.
This is not the original projected Pegasos algorithm.
"""
from __future__ import annotations
import json
import time
from pathlib import Path
import numpy as np


def hinge_terms(margins):
    """返回逐样本 hinge 损失及其对 margin 的子梯度；margin=1 时取 0。"""
    # BEGIN TODO 2
    # 同时返回逐样本 hinge 损失和 margin 子梯度；margin=1 时子梯度取 0。
    raise NotImplementedError("请完成 TODO 2")
    # END TODO 2


def logistic_terms(margins):
    """选做 A1：返回逐样本 logistic 损失与 margin 导数，极端输入仍稳定。"""
    # BEGIN OPTIONAL A1
    # 返回稳定的逐样本 logistic 损失和 margin 导数；极大正负输入也应有限。
    raise NotImplementedError("请完成 OPTIONAL A1")
    # END OPTIONAL A1


def rbf_kernel(x, z, gamma):
    """选做 A2：K[i,j]=exp(-gamma*||x[i]-z[j]||^2)，结果形状 [n_x,n_z]。"""
    if gamma <= 0:
        raise ValueError('gamma must be positive')
    x, z = np.asarray(x, dtype=np.float64), np.asarray(z, dtype=np.float64)
    # BEGIN OPTIONAL A2
    # 用广播或矩阵运算构造 [n_x,n_z] 的 RBF 核矩阵；处理距离的舍入误差。
    raise NotImplementedError("请完成 OPTIONAL A2")
    # END OPTIONAL A2


class LinearClassifier:
    def __init__(self, features=7, regularization=0.01, lr=0.05, epochs=150,
                 batch_size=64, loss='hinge', seed=42):
        if features < 1 or regularization <= 0 or lr <= 0 or epochs < 1 or batch_size < 1:
            raise ValueError('features, regularization, lr, epochs and batch_size must be positive')
        if loss not in ('hinge', 'logistic'):
            raise ValueError('loss must be hinge or logistic')
        self.features, self.regularization, self.lr = features, regularization, lr
        self.epochs, self.batch_size, self.loss, self.seed = epochs, batch_size, loss, seed
        self.w = np.zeros(features, dtype=np.float64)
        self.b = 0.0

    def decision_function(self, x):
        """必做 1：返回每个样本的线性分数，[m,d] -> [m]。"""
        # BEGIN TODO 1
        # 返回 X 与当前权重、偏置的批量线性分数，形状 [m]。
        raise NotImplementedError("请完成 TODO 1")
        # END TODO 1

    def terms(self, margins):
        return hinge_terms(margins) if self.loss == 'hinge' else logistic_terms(margins)

    def objective(self, x, y):
        """必做 3：平均数据损失 + L2 权重惩罚，返回 Python float。"""
        losses, _ = self.terms(y * self.decision_function(x))
        # BEGIN TODO 3
        # 返回平均数据损失加 L2 权重正则项，使用 Python float；不正则化偏置。
        raise NotImplementedError("请完成 TODO 3")
        # END TODO 3

    def gradients(self, x, y):
        """必做 4：dw=[d], db=scalar；按真实批次大小平均一次，b 不正则化。"""
        _, margin_slopes = self.terms(y * self.decision_function(x))
        # BEGIN TODO 4
        # 先经 margin 对分数应用链式法则，再计算 dw、db；按 len(y) 平均一次。
        raise NotImplementedError("请完成 TODO 4")
        # END TODO 4

    def learning_rate(self, epoch):
        """必做 5：按 epoch 从 1 开始，学习率 lr/sqrt(epoch)。"""
        # BEGIN TODO 5
        # epoch 从 1 开始，返回 lr/sqrt(epoch)。
        raise NotImplementedError("请完成 TODO 5")
        # END TODO 5

    def update(self, dw, db, eta):
        """必做 6：做一次 SGD 更新；梯度已平均、已包含 L2。"""
        # BEGIN TODO 6
        # 使用已经平均且包含 L2 的 dw、db 做一次 SGD 更新。
        raise NotImplementedError("请完成 TODO 6")
        # END TODO 6

    def predict(self, x):
        # 分数恰为 0 时统一预测为 +1。
        return np.where(self.decision_function(x) >= 0.0, 1, -1)

    def data_loss(self, x, y):
        return float(self.terms(y * self.decision_function(x))[0].mean())

    def fit(self, x, y, validation_x, validation_y, verbose=False):
        start = time.perf_counter()
        x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
        if x.ndim != 2 or x.shape != (len(y), self.features) or y.ndim != 1:
            raise ValueError('X must be [n,d] and y [n]')
        if not np.isin(y, [-1, 1]).all() or not np.isfinite(x).all():
            raise ValueError('Expected finite X and labels in {-1,+1}')
        rng = np.random.default_rng(self.seed)
        self.history, best_loss = [], np.inf
        self.best_epoch = None
        for epoch in range(1, self.epochs + 1):
            eta = self.learning_rate(epoch)
            order = rng.permutation(len(y))
            for offset in range(0, len(order), self.batch_size):
                idx = order[offset:offset + self.batch_size]
                dw, db = self.gradients(x[idx], y[idx])
                if not np.isfinite(dw).all() or not np.isfinite(db):
                    raise FloatingPointError(f'Nonfinite gradient at epoch={epoch}; preserve the failed run')
                self.update(dw, db, eta)
            row = {'epoch': epoch, 'learning_rate': float(eta),
                   'train_objective': self.objective(x, y),
                   'train_data_loss': self.data_loss(x, y),
                   'train_accuracy': float(np.mean(self.predict(x) == y)),
                   'validation_data_loss': self.data_loss(validation_x, validation_y),
                   'validation_accuracy': float(np.mean(self.predict(validation_x) == validation_y)),
                   'weight_norm': float(np.linalg.norm(self.w)),
                   'train_margin_violation_fraction': float(np.mean(y * self.decision_function(x) < 1.0))}
            if not all(np.isfinite(v) for v in row.values()):
                raise FloatingPointError(f'Nonfinite metric at epoch={epoch}')
            self.history.append(row)
            # 不用测试集选轮；此处也不把正则项加入验证损失。
            if row['validation_data_loss'] < best_loss:
                best_loss, self.best_epoch = row['validation_data_loss'], epoch
                best_w, best_b = self.w.copy(), self.b
            if verbose and (epoch == 1 or epoch % 50 == 0 or epoch == self.epochs):
                print(f'epoch={epoch:03d} lr={eta:.5f} train_loss={row["train_data_loss"]:.4f} '
                      f'val_loss={row["validation_data_loss"]:.4f} val_acc={row["validation_accuracy"]:.4f}', flush=True)
        self.last_state = (self.w.copy(), self.b)
        self.w, self.b = best_w, best_b
        self.training_seconds = time.perf_counter() - start
        return self

    def save(self, path):
        meta = {'kind': 'manual', **{k: getattr(self, k) for k in
                ('features', 'regularization', 'lr', 'epochs', 'batch_size', 'loss', 'seed')}}
        np.savez_compressed(path, metadata=np.array(json.dumps(meta)), w=self.w, b=np.array(self.b))


class ReferenceSVM:
    """教师提供的成熟库对照；RBF 使用选做 A2 的核矩阵。"""
    def __init__(self, kernel='linear', C=1.0, gamma=1.0):
        self.kernel, self.C, self.gamma = kernel, C, gamma
        self.loss = 'hinge'
        self.best_epoch = None

    def fit(self, x, y):
        from sklearn.svm import SVC
        start = time.perf_counter()
        self.estimator = SVC(C=self.C, kernel='linear' if self.kernel == 'linear' else 'precomputed', tol=1e-7)
        self.estimator.fit(x if self.kernel == 'linear' else rbf_kernel(x, x, self.gamma), y)
        if self.kernel == 'linear':
            self.w = self.estimator.coef_[0].copy()
        else:
            self.support_x = x[self.estimator.support_].copy()
            self.dual_coef = self.estimator.dual_coef_[0].copy()
        self.b = float(self.estimator.intercept_[0])
        self.n_support = self.estimator.n_support_.tolist()
        self.training_seconds = time.perf_counter() - start
        return self

    def decision_function(self, x):
        if self.kernel == 'linear':
            return x @ self.w + self.b
        return rbf_kernel(x, self.support_x, self.gamma) @ self.dual_coef + self.b

    def predict(self, x):
        return np.where(self.decision_function(x) >= 0.0, 1, -1)

    def data_loss(self, x, y):
        return float(hinge_terms(y * self.decision_function(x))[0].mean())

    def save(self, path):
        meta = {'kind': 'reference', 'kernel': self.kernel, 'C': self.C, 'gamma': self.gamma}
        arrays = {'w': self.w} if self.kernel == 'linear' else {'support_x': self.support_x, 'dual_coef': self.dual_coef}
        np.savez_compressed(path, metadata=np.array(json.dumps(meta)), b=np.array(self.b), **arrays)


def load_model(path):
    with np.load(Path(path), allow_pickle=False) as z:
        meta = json.loads(str(z['metadata']))
        kind = meta.pop('kind')
        model = LinearClassifier(**meta) if kind == 'manual' else ReferenceSVM(**meta)
        model.b = float(z['b'])
        if kind == 'manual' or model.kernel == 'linear':
            model.w = z['w'].copy()
        else:
            model.support_x, model.dual_coef = z['support_x'].copy(), z['dual_coef'].copy()
        return model

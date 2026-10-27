"""Mini-batch fully connected network with explicit NumPy backpropagation.

Convention: X is [batch, input], W is [input, output], b is [output].
Objective: mean softmax cross entropy + l2 / 2 * sum(W ** 2).
Biases are not regularized. No automatic differentiation is used.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def sigmoid(z):
    """Stable sigmoid, including large positive/negative arguments."""
    out = np.empty_like(z)
    positive = z >= 0
    out[positive] = 1 / (1 + np.exp(-z[positive]))
    exp_z = np.exp(z[~positive])
    out[~positive] = exp_z / (1 + exp_z)
    return out


def softmax(logits):
    shifted = logits - logits.max(axis=1, keepdims=True)
    exponentials = np.exp(shifted)
    return exponentials / exponentials.sum(axis=1, keepdims=True)


def cross_entropy(logits, labels):
    """Mean cross entropy computed from logits, without log(0)."""
    shifted = logits - logits.max(axis=1, keepdims=True)
    # logits: [m, C]；labels: [m]，为 0 至 C-1 的整数标签。
    # shifted 已减去每行最大值；从它计算每个样本的负对数似然，返回平均值。
    # BEGIN TODO 1
    # 必做：实现数值稳定的多分类平均交叉熵，返回 Python float。
    raise NotImplementedError("请完成 TODO 1")
    # END TODO 1


class Network:
    def __init__(self, sizes=(784, 64, 24), activation='relu', l2=0.0,
                 seed=42, dtype=np.float32):
        if len(sizes) < 2 or any(int(size) != size or size <= 0 for size in sizes):
            raise ValueError('sizes must contain at least two positive integer dimensions')
        if activation not in ('relu', 'sigmoid'):
            raise ValueError('activation must be relu or sigmoid')
        if l2 < 0:
            raise ValueError('l2 must be nonnegative')
        self.sizes = tuple(int(size) for size in sizes)
        self.activation = activation
        self.l2 = float(l2)
        self.dtype = np.dtype(dtype)
        rng = np.random.default_rng(seed)
        self.weights, self.biases = [], []
        for index, (fan_in, fan_out) in enumerate(zip(self.sizes[:-1], self.sizes[1:])):
            # He initialization for ReLU hidden layers; Xavier-style for output/Sigmoid.
            gain = 2.0 if activation == 'relu' and index < len(self.sizes) - 2 else 1.0
            self.weights.append((rng.standard_normal((fan_in, fan_out)) * np.sqrt(gain / fan_in)).astype(self.dtype))
            self.biases.append(np.zeros(fan_out, dtype=self.dtype))

    def forward(self, x, return_cache=False):
        """批量前向传播。X=[m,d]，W=[d,h]，b=[h]。

        缓存 activations 包含输入及各隐藏层输出，不含输出层概率；
        preactivations 包含每层线性输出 z（包括输出层）。
        训练和预测共用此函数，无须重复实现两遍。
        """
        activations, preactivations = [x], []
        current = x
        for index, (weight, bias) in enumerate(zip(self.weights, self.biases)):
            # 用 current 计算本层 z；仅在隐藏层对 z 使用 ReLU 或 Sigmoid。
            # 输出层保持线性输出，统一由循环后的 Softmax 处理。
            # BEGIN TODO 2
            # 必做：补全线性变换，以及隐藏层 current 的激活计算。
            raise NotImplementedError("请完成 TODO 2")
            # END TODO 2
            preactivations.append(z)
            if index < len(self.weights) - 1:
                activations.append(current)
        probabilities = softmax(preactivations[-1])
        if return_cache:
            return probabilities, (activations, preactivations)
        return probabilities

    def regularization_loss(self):
        return self.l2 / 2 * sum(float(np.sum(weight * weight, dtype=np.float64)) for weight in self.weights)

    def backward(self, x, y):
        """返回 (总损失, 各层权重梯度列表, 各层偏置梯度列表)。

        总损失 = 批次平均交叉熵 + l2/2 * sum(W**2)。
        delta 表示损失对当前层线性输出 z 的梯度。
        """
        if len(y) == 0:
            raise ValueError('empty mini-batch')
        probabilities, (activations, preactivations) = self.forward(x, return_cache=True)
        loss = cross_entropy(preactivations[-1], y) + self.regularization_loss()
        # 使用真实批次大小 len(y)，最后一个批次可能小于 batch_size。
        # 不要原地修改 probabilities；批次平均仅在此处完成一次。
        # BEGIN TODO 3
        # 必做：计算 Softmax + 平均交叉熵的输出层 delta，形状 [m,C]。
        raise NotImplementedError("请完成 TODO 3")
        # END TODO 3
        grad_weights = [None] * len(self.weights)
        grad_biases = [None] * len(self.biases)
        for layer in range(len(self.weights) - 1, -1, -1):
            # activations[layer] 为本层输入；delta 已包含批次平均系数。
            # BEGIN TODO 4
            # 必做：计算 grad_weights[layer]，包含 L2 项；结果形状与权重一致。
            raise NotImplementedError("请完成 TODO 4")
            # END TODO 4
            # BEGIN TODO 5
            # 必做：沿样本维汇总 grad_biases[layer]；偏置不做 L2 正则化。
            raise NotImplementedError("请完成 TODO 5")
            # END TODO 5
            if layer > 0:
                # 先把误差传播至上一隐藏层，再乘相应激活函数的导数。
                # ReLU 在 z=0 处导数约定为 0；Sigmoid 可利用缓存激活值。
                # BEGIN TODO 6
                # 必做：更新上一隐藏层的 delta，分别支持 ReLU 与 Sigmoid。
                raise NotImplementedError("请完成 TODO 6")
                # END TODO 6
        return loss, grad_weights, grad_biases

    def update(self, grad_weights, grad_biases, learning_rate):
        """使用已计算好的平均梯度进行一次 SGD 更新。"""
        for layer in range(len(self.weights)):
            # 此处不要再次除以批次大小，也不要再次添加 L2 项。
            # BEGIN TODO 7
            # 必做：更新本层权重和偏置，保持原形状。
            raise NotImplementedError("请完成 TODO 7")
            # END TODO 7

    def evaluate(self, x, y, batch_size=1024):
        total_loss, correct = 0.0, 0
        for start in range(0, len(y), batch_size):
            xb, yb = x[start:start + batch_size], y[start:start + batch_size]
            probabilities, (_, zs) = self.forward(xb, return_cache=True)
            total_loss += len(yb) * cross_entropy(zs[-1], yb)
            correct += int(np.sum(probabilities.argmax(axis=1) == yb))
        return {'cross_entropy': total_loss / len(y),
                'objective': total_loss / len(y) + self.regularization_loss(),
                'accuracy': correct / len(y)}

    def predict_proba(self, x, batch_size=1024):
        return np.concatenate([self.forward(x[start:start + batch_size])
                               for start in range(0, len(x), batch_size)])

    def save(self, path):
        metadata = {'sizes': self.sizes, 'activation': self.activation,
                    'l2': self.l2, 'dtype': self.dtype.name}
        arrays = {'metadata': np.array(json.dumps(metadata))}
        arrays.update({f'W{i}': weight for i, weight in enumerate(self.weights)})
        arrays.update({f'b{i}': bias for i, bias in enumerate(self.biases)})
        np.savez_compressed(path, **arrays)

    @classmethod
    def load(cls, path):
        with np.load(Path(path), allow_pickle=False) as archive:
            metadata = json.loads(str(archive['metadata']))
            network = cls(**metadata)
            network.weights = [archive[f'W{i}'].copy() for i in range(len(network.weights))]
            network.biases = [archive[f'b{i}'].copy() for i in range(len(network.biases))]
        return network

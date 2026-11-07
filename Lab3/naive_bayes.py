"""5处必做、附加A1伯努利、附加A2词语贡献解释。

大家只需修改本文件中的 TODO / OPTIONAL 区域。
X=[N,V] 是非负词频稀疏矩阵，C由训练标签确定，不能把150写死。
可以使用 NumPy/SciPy；核心训练和预测须自行实现。
"""

from pathlib import Path

import numpy as np
from scipy import sparse
from scipy.special import logsumexp


class MultinomialNaiveBayes:
    model_kind = "multinomial"

    def __init__(self, alpha=1.0):
        if not np.isfinite(alpha) or alpha <= 0:
            raise ValueError("alpha 必须为有限正数")
        self.alpha = float(alpha)

    @staticmethod
    def _matrix(x):
        x = sparse.csr_matrix(x, dtype=np.float64)
        if x.shape[1] == 0 or not np.isfinite(x.data).all() or (x.data < 0).any():
            raise ValueError("X 应具有至少一个特征，且所有元素有限、非负")
        return x

    def _fit_inputs(self, x, y):
        x = self._matrix(x)
        y = np.asarray(y)
        if y.ndim != 1 or len(y) != x.shape[0] or len(y) == 0:
            raise ValueError("y应为与X对齐的非空一维标签数组")
        self.classes_, encoded = np.unique(y, return_inverse=True)
        n, vocabulary_size = x.shape
        number_of_classes = len(self.classes_)
        indicator = sparse.csr_matrix(
            (np.ones(n), (np.arange(n), encoded)),
            shape=(n, number_of_classes),
        )
        return x, encoded, n, vocabulary_size, number_of_classes, indicator

    def fit(self, x, y):
        x, encoded, n, vocabulary_size, number_of_classes, indicator = self._fit_inputs(x, y)

        # TODO 1：统计各类文档数，计算对数先验。
        # encoded=[N]，其值为0到C-1；n为训练文档数。
        # 保存 class_count_、class_log_prior_，形状均为[C]。
        # 注意：先验按文档数估计，不能用总词数，不能写死为1/150。
        # BEGIN TODO 1
        raise NotImplementedError("请完成 TODO 1")
        # END TODO 1

        # TODO 2：累加同类文档的词频。
        # x=[N,V]；indicator=[N,C]，是已提供的类别指示矩阵。
        # 保存 feature_count_=[C,V]，可按类别求和，也可使用矩阵运算。
        # 允许将[C,V]结果转为稠密矩阵，不要展开整个[N,V]文档矩阵。
        # BEGIN TODO 2
        raise NotImplementedError("请完成 TODO 2")
        # END TODO 2

        # TODO 3：实现加alpha平滑、按类别归一化，得到对数条件概率。
        # 输入 feature_count_=[C,V]，平滑系数 self.alpha。
        # 保存 feature_log_prob_=[C,V]，所有值应有限。
        # 每类概率之和为1：exp(feature_log_prob_).sum(axis=1)应接近1。
        # BEGIN TODO 3
        raise NotImplementedError("请完成 TODO 3")
        # END TODO 3

        self.n_features_in_ = vocabulary_size
        return self

    def _prediction_inputs(self, x):
        if not hasattr(self, "feature_log_prob_"):
            raise RuntimeError("请先fit模型")
        x = self._matrix(x)
        if x.shape[1] != self.n_features_in_:
            raise ValueError("预测词表维度与训练不一致")
        return x

    def joint_log_scores(self, x):
        x = self._prediction_inputs(x)
        # TODO 4：计算每个输入样本属于每一类的对数联合得分。
        # x=[B,V]，feature_log_prob_=[C,V]，class_log_prior_=[C]。
        # 返回普通NumPy数组 scores=[B,C]，同时包含先验和词频贡献。
        # 这是未归一化得分，不是概率；没有任何已知词时应只剩对数先验。
        # BEGIN TODO 4
        raise NotImplementedError("请完成 TODO 4")
        # END TODO 4

    def predict(self, x):
        scores = self.joint_log_scores(x)
        # TODO 5：逐样本选得分最高的类别，返回形状[B]的类别标签。
        # classes_ 保存列索引对应的真实标签，标签未必是连续整数。
        # 得分相同，按classes_顺序取第一个。
        # BEGIN TODO 5
        raise NotImplementedError("请完成 TODO 5")
        # END TODO 5

    def predict_log_proba(self, x):
        # 框架提供：稳定归一化。模型概率不等于已校准的正确率。
        scores = self.joint_log_scores(x)
        return scores - logsumexp(scores, axis=1, keepdims=True)

    def save(self, path):
        values = {
            "model_kind": np.array(self.model_kind),
            "alpha": np.array(self.alpha),
            "classes": self.classes_,
            "class_count": self.class_count_,
            "class_log_prior": self.class_log_prior_,
            "feature_count": self.feature_count_,
            "feature_log_prob": self.feature_log_prob_,
        }
        if self.model_kind == "bernoulli":
            values["feature_neg_log_prob"] = self.feature_neg_log_prob_
        np.savez_compressed(Path(path), **values)

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as data:
            kind = str(data["model_kind"]) if "model_kind" in data else "multinomial"
            model_class = BernoulliNaiveBayes if kind == "bernoulli" else MultinomialNaiveBayes
            model = model_class(float(data["alpha"]))
            for name in ("classes", "class_count", "class_log_prior", "feature_count", "feature_log_prob"):
                setattr(model, name + "_", data[name].copy())
            if kind == "bernoulli":
                model.feature_neg_log_prob_ = data["feature_neg_log_prob"].copy()
        model.n_features_in_ = model.feature_log_prob_.shape[1]
        return model

    def explain_pair(self, x_row, positive_class, negative_class, feature_names, top_n=10):
        """附加A2：解释两个类别的得分差；仅用于多项式模型。"""
        if self.model_kind != "multinomial":
            raise ValueError("附加A2限定在多项式模型上完成")
        x_row = self._prediction_inputs(x_row)
        if x_row.shape[0] != 1:
            raise ValueError("解释输入应为[1,V]")
        positive = int(np.flatnonzero(self.classes_ == positive_class)[0])
        negative = int(np.flatnonzero(self.classes_ == negative_class)[0])
        # A2要求：计算“positive类别得分 - negative类别得分”的分解。
        # coefficients=[V]：每个特征对应的对数条件概率差。
        # contributions=[nnz]：该样本非零特征的得分差贡献，要乘词频。
        # prior_gap：两个类别的对数先验差，Python float。
        # 后续排序、输出和求和一致性检查已提供。
        # BEGIN OPTIONAL A2
        raise NotImplementedError("请完成 OPTIONAL A2")
        # END OPTIONAL A2

        order = np.argsort(-np.abs(contributions), kind="stable")[:top_n]
        terms = [{"feature": str(feature_names[x_row.indices[j]]), "count": float(x_row.data[j]), "contribution_to_score_gap": float(contributions[j])} for j in order]
        scores = self.joint_log_scores(x_row)[0]
        gap = float(scores[positive] - scores[negative])
        reconstructed = prior_gap + float(contributions.sum())
        if not np.isclose(gap, reconstructed, atol=1e-10, rtol=1e-10):
            raise AssertionError("词语贡献与先验差之和不等于分类得分差")
        return {"score_gap": gap, "prior_gap": prior_gap, "all_feature_contributions_sum": float(contributions.sum()), "top_features_by_absolute_contribution": terms}


class BernoulliNaiveBayes(MultinomialNaiveBayes):
    """附加A1：完整实现伯努利模型。两块均完成后用 --model bernoulli。"""

    model_kind = "bernoulli"

    def fit(self, x, y):
        x, encoded, n, vocabulary_size, number_of_classes, indicator = self._fit_inputs(x, y)
        # A1_FIT：从词频矩阵得到“出现/未出现”，统计每类文档数和词出现文档数。
        # 保存 class_count_、class_log_prior_：[C]。
        # 保存 feature_count_：[C,V]，记录出现文档数，不是累计词频。
        # 按伯努利加alpha平滑估计 P(word出现|class)。
        # 保存 feature_log_prob_=log(P)，feature_neg_log_prob_=log(1-P)，均[C,V]。
        # 每个特征是独立伯努利变量，不沿词表归一化；分母应考虑两个取值。
        # BEGIN OPTIONAL A1_FIT
        raise NotImplementedError("请完成 OPTIONAL A1_FIT")
        # END OPTIONAL A1_FIT

        self.n_features_in_ = vocabulary_size
        return self

    def joint_log_scores(self, x):
        x = self._prediction_inputs(x)
        # A1_SCORE：返回[B,C]的伯努利对数联合得分。
        # 预测时也要二值化，重复出现同一个词不能重复加分。
        # 必须同时计入出现词的log(P)和未出现词的log(1-P)，并加对数先验。
        # 可通过矩阵变形避免创建[B,V]的稠密“1-X”矩阵。
        # BEGIN OPTIONAL A1_SCORE
        raise NotImplementedError("请完成 OPTIONAL A1_SCORE")
        # END OPTIONAL A1_SCORE

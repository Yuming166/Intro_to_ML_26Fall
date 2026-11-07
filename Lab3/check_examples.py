"""公开小例自检。完成必做：python check_examples.py
只核对fit：python check_examples.py --stage fit
附加题：python check_examples.py --optional bernoulli / --optional explain
"""

import argparse
import sys

import numpy as np
from scipy import sparse

from naive_bayes import BernoulliNaiveBayes, MultinomialNaiveBayes


def check(stage="all", optional=None):
    x = sparse.csr_matrix([[2, 1, 0], [0, 1, 1], [0, 0, 2]], dtype=float)
    # 非连续标签用于检查predict是否正确映射标签；两类文档数不均衡。
    y = np.array([2, 2, 8])
    probe = sparse.csr_matrix([[1, 0, 1]], dtype=float)
    model = MultinomialNaiveBayes(alpha=1).fit(x, y)
    np.testing.assert_allclose(model.class_count_, [2, 1])
    np.testing.assert_allclose(np.exp(model.class_log_prior_), [2 / 3, 1 / 3])
    np.testing.assert_allclose(model.feature_count_, [[2, 2, 1], [0, 0, 2]])
    np.testing.assert_allclose(np.exp(model.feature_log_prob_), [[3 / 8, 3 / 8, 2 / 8], [1 / 5, 1 / 5, 3 / 5]])
    print("PASS：类别计数、先验、词频统计和平滑概率。")
    if stage == "fit" and optional is None:
        return
    expected = np.log([1 / 16, 1 / 25])
    np.testing.assert_allclose(model.joint_log_scores(probe)[0], expected)
    np.testing.assert_array_equal(model.predict(probe), [2])
    empty = sparse.csr_matrix((1, 3))
    np.testing.assert_allclose(model.joint_log_scores(empty)[0], model.class_log_prior_)
    np.testing.assert_array_equal(model.predict(empty), [2])
    print("PASS：分类得分、真实标签映射、没有已知词时的处理。")
    if optional == "bernoulli":
        binary_model = BernoulliNaiveBayes(alpha=1).fit(x, y)
        np.testing.assert_allclose(binary_model.feature_count_, [[1, 2, 1], [0, 0, 1]])
        np.testing.assert_allclose(np.exp(binary_model.feature_log_prob_), [[1 / 2, 3 / 4, 1 / 2], [1 / 3, 1 / 3, 2 / 3]])
        expected_binary = np.log([1 / 24, 4 / 81])
        np.testing.assert_allclose(binary_model.joint_log_scores(probe)[0], expected_binary)
        np.testing.assert_array_equal(binary_model.predict(probe), [8])
        np.testing.assert_allclose(binary_model.joint_log_scores(sparse.csr_matrix([[3, 0, 2]])), binary_model.joint_log_scores(probe))
        print("PASS：伯努利文档频次、出现/未出现贡献、重复词二值化。")
    if optional == "explain":
        result = model.explain_pair(probe, 2, 8, np.array(["a", "b", "c"]))
        np.testing.assert_allclose(result["score_gap"], expected[0] - expected[1])
        np.testing.assert_allclose(result["prior_gap"] + result["all_feature_contributions_sum"], result["score_gap"])
        print("PASS：词语贡献与先验差重建分类得分差。")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["fit", "all"], default="all")
    parser.add_argument("--optional", choices=["bernoulli", "explain"], default=None)
    args = parser.parse_args()
    try:
        check(args.stage, args.optional)
    except NotImplementedError as error:
        print(f"尚未完成：{error}", file=sys.stderr)
        return 2
    except AssertionError as error:
        print(f"数值核对失败，请检查最近一项计算：\n{error}", file=sys.stderr)
        return 1
    print("本次小例核对全部通过。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

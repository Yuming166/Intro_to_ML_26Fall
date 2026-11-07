"""Task3：完整150类意图分类。默认只评价训练/验证，不打开测试集。

python main.py
python main.py --alpha 0.1 --final-test
python main.py --model bernoulli            # 需要完成附加A1
python main.py --explain                    # 需要完成附加A2
"""

import argparse
import csv
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import platform
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy
import sklearn
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from threadpoolctl import threadpool_limits

from clinc_loader import load_data, no_train_overlap_mask, sha256
from naive_bayes import BernoulliNaiveBayes, MultinomialNaiveBayes
from text_features import TextFeatures, matrix_info


ROOT = Path(__file__).resolve().parent


def write_json(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def write_csv(path, rows, columns):
    with Path(path).open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def measure(y, predicted, number_of_classes):
    return {"n": int(len(y)), "correct": int(np.count_nonzero(y == predicted)), "accuracy": float(accuracy_score(y, predicted)), "macro_f1": float(f1_score(y, predicted, labels=np.arange(number_of_classes), average="macro", zero_division=0))}


def evaluate(output, name, split, x, model, class_names, train_texts, explain=False, feature_names=None):
    # 必须经过学生实现的predict，不能由框架绕过TODO 5。
    predicted = np.asarray(model.predict(x))
    if predicted.shape != split.y.shape:
        raise ValueError("predict应返回每个样本对应的一个类别标签")
    scores = model.joint_log_scores(x)
    if scores.shape != (len(split.y), len(class_names)) or not np.isfinite(scores).all():
        raise ValueError("joint_log_scores应返回有限的[B,C]数组")
    official = measure(split.y, predicted, len(class_names))
    clean_mask = no_train_overlap_mask(train_texts, split.texts)
    clean = measure(split.y[clean_mask], predicted[clean_mask], len(class_names))
    report = classification_report(split.y, predicted, labels=np.arange(len(class_names)), target_names=class_names, output_dict=True, zero_division=0)
    per_class = [{"label": label, "precision": report[label]["precision"], "recall": report[label]["recall"], "f1": report[label]["f1-score"], "support": int(report[label]["support"])} for label in class_names]
    write_csv(output / f"{name}_per_class.csv", per_class, ["label", "precision", "recall", "f1", "support"])
    cm = confusion_matrix(split.y, predicted, labels=np.arange(len(class_names)))
    pairs = [{"true_label": class_names[i], "predicted_label": class_names[j], "count": int(cm[i, j])} for i in range(len(class_names)) for j in range(len(class_names)) if i != j and cm[i, j]]
    pairs.sort(key=lambda row: (-row["count"], row["true_label"], row["predicted_label"]))
    write_csv(output / f"{name}_confusion_pairs.csv", pairs, ["true_label", "predicted_label", "count"])
    order = np.argsort(-scores, axis=1, kind="stable")[:, :3]
    rows = [{"row_index": i, "text": text, "true_label": class_names[int(split.y[i])], "predicted_label": class_names[int(predicted[i])], "correct": bool(split.y[i] == predicted[i]), "top2_label": class_names[int(model.classes_[order[i, 1]])], "top3_label": class_names[int(model.classes_[order[i, 2]])], "no_train_text_overlap": bool(clean_mask[i])} for i, text in enumerate(split.texts)]
    columns = ["row_index", "text", "true_label", "predicted_label", "correct", "top2_label", "top3_label", "no_train_text_overlap"]
    write_csv(output / f"{name}_predictions.csv", rows, columns)
    write_csv(output / f"{name}_errors.csv", [row for row in rows if not row["correct"]], columns)
    if pairs:
        top = pairs[:15]
        fig, axis = plt.subplots(figsize=(12, 7))
        axis.barh(np.arange(len(top)), [row["count"] for row in top], color="#4474a4")
        axis.set_yticks(np.arange(len(top)), [f"{row['true_label']} -> {row['predicted_label']}" for row in top])
        axis.invert_yaxis()
        axis.set(xlabel="Number of errors", title=f"Most frequent {name} confusions")
        fig.tight_layout()
        fig.savefig(output / f"{name}_confusions.png", dpi=140)
        plt.close(fig)
    if explain:
        indices = np.flatnonzero(predicted == split.y)[:3].tolist() + np.flatnonzero(predicted != split.y)[:3].tolist()
        explanations = []
        for i in indices:
            other = int(split.y[i]) if predicted[i] != split.y[i] else int(model.classes_[order[i, 1]])
            detail = model.explain_pair(x[i], int(predicted[i]), other, feature_names)
            explanations.append({"row_index": i, "text": split.texts[i], "true_label": class_names[int(split.y[i])], "predicted_label": class_names[int(predicted[i])], "compared_with_label": class_names[other], **detail})
        write_json(output / f"{name}_word_explanations.json", explanations)
    return {"official": official, "excluding_train_text_overlap": clean, "feature_matrix": matrix_info(x)}


def run(args, output):
    started = time.perf_counter()
    splits, names, audit, folder = load_data(args.data_dir, include_test=args.final_test)
    config = {"model": args.model, "alpha": args.alpha, "ngram_max": args.ngram_max, "stopwords": args.stopwords, "explain": args.explain, "final_test_requested": args.final_test, "class_count": len(names)}
    # 先记录配置，再做训练与最终评价，不自动搜索测试分数。
    write_json(output / "config_before_training.json", config)
    write_json(output / "data_audit.json", audit)
    write_json(output / "environment.json", {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__, "scikit_learn": sklearn.__version__, "threads": args.threads, "source_hashes": {name: sha256(ROOT / name) for name in ["main.py", "naive_bayes.py", "text_features.py", "clinc_loader.py"]}})
    stopwords = None
    if args.stopwords == "english":
        metadata = json.loads((folder / "SOURCE.json").read_text(encoding="utf-8"))
        if sha256(folder / "english_stopwords.txt") != metadata["stopwords_sha256"]:
            raise ValueError("冻结的停用词表发生变化")
        stopwords = (folder / "english_stopwords.txt").read_text(encoding="utf-8").splitlines()
    features = TextFeatures(args.ngram_max, stopwords)
    train_x = features.fit_transform(splits["train"].texts)
    val_x = features.transform(splits["val"].texts)
    classifier = BernoulliNaiveBayes if args.model == "bernoulli" else MultinomialNaiveBayes
    fit_started = time.perf_counter()
    model = classifier(args.alpha).fit(train_x, splits["train"].y)
    fit_seconds = time.perf_counter() - fit_started
    train_predicted = model.predict(train_x)
    train_metrics = measure(splits["train"].y, train_predicted, len(names))
    validation = evaluate(output, "validation", splits["val"], val_x, model, names, splits["train"].texts, args.explain, features.names())
    model.save(output / "model.npz")
    features.save(output / "features.json")
    write_json(output / "classes.json", names)
    results = {"config": config, "train": train_metrics, "validation": validation, "fit_seconds": fit_seconds, "test_evaluated": False, "oos_used": False}
    write_json(output / "metrics.json", results)
    v = validation["official"]
    print(f"验证集: {v['correct']}/{v['n']}，准确率={v['accuracy']:.2%}，Macro-F1={v['macro_f1']:.4f}", flush=True)
    if args.final_test:
        write_json(output / "config_before_test.json", {**config, "validation_metrics": v, "test_scores_computed": False})
        test_x = features.transform(splits["test"].texts)
        results["test"] = evaluate(output, "test", splits["test"], test_x, model, names, splits["train"].texts, args.explain, features.names())
        results["test_evaluated"] = True
        t = results["test"]["official"]
        print(f"测试集: {t['correct']}/{t['n']}，准确率={t['accuracy']:.2%}，Macro-F1={t['macro_f1']:.4f}", flush=True)
    else:
        print("本次仅作开发，未打开测试集。确定配置后再使用 --final-test。", flush=True)
    results["elapsed_seconds"] = time.perf_counter() - started
    write_json(output / "metrics.json", results)
    write_json(output / "COMPLETION.json", {"status": "COMPLETE", "config": config, "test_evaluated": args.final_test, "oos_used": False, "elapsed_seconds": results["elapsed_seconds"]})
    print(f"输出目录: {output}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", default=None, help="默认读取脚本旁data/中的CSV")
    parser.add_argument("--output", default=None, help="新的空输出目录；默认脚本旁output/时间戳")
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--ngram-max", type=int, choices=[1, 2], default=1)
    parser.add_argument("--stopwords", choices=["none", "english"], default="none")
    parser.add_argument("--model", choices=["multinomial", "bernoulli"], default="multinomial")
    parser.add_argument("--final-test", action="store_true", help="确定配置后显式启用最终测试")
    parser.add_argument("--explain", action="store_true", help="启用附加A2词语贡献解释")
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    if not np.isfinite(args.alpha) or args.alpha <= 0 or args.threads < 1:
        parser.error("alpha和threads必须为正数")
    if args.model == "bernoulli" and args.explain:
        parser.error("A2只在多项式模型上运行，请分别运行两个附加题")
    name = datetime.now(timezone(timedelta(hours=8))).strftime("%Y%m%d_%H%M%S_%f")
    output = Path(args.output).expanduser().resolve() if args.output else ROOT / "output" / name
    if output.exists() and any(output.iterdir()):
        parser.error("输出目录不为空，请使用新目录，保留旧实验记录")
    output.mkdir(parents=True, exist_ok=True)
    with threadpool_limits(limits=args.threads):
        try:
            run(args, output)
        except NotImplementedError as error:
            write_json(output / "INCOMPLETE.json", {"status": "INCOMPLETE", "message": str(error)})
            print(f"尚未完成：{error}\n请按naive_bayes.py中的要求补全后重新运行。", file=sys.stderr)
            return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

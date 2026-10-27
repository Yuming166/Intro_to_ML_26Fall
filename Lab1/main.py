"""先完成 network.py 的必做题，再尝试 augmentation.py 的选做题。"""
from __future__ import annotations

import argparse
import ast
import csv
import importlib.metadata
import json
import os
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from threadpoolctl import threadpool_limits

from network import Network
from sign_mnist_loader import load_data, sha256, resolve_data_dir
from augmentation import augment_batch, shift_batch


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')


def plots(output, history, split, probabilities, letters, normalization='center'):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    epochs = [row['epoch'] for row in history]
    curves = [('train', 'Train'), ('validation', 'Validation')]
    if 'stress_validation_accuracy' in history[0]:
        curves.append(('stress_validation', 'Shifted validation'))
    for name, label in curves:
        axes[0].plot(epochs, [row[name + '_cross_entropy'] for row in history], label=label)
        axes[1].plot(epochs, [row[name + '_accuracy'] for row in history], label=label)
    axes[0].set(ylabel='Mean cross entropy', xlabel='Epoch', title='Learning curves: data loss')
    axes[1].set(ylabel='Accuracy', xlabel='Epoch', title='Learning curves: accuracy', ylim=(0, 1.02))
    for axis in axes:
        axis.grid(alpha=.25)
        axis.legend()
    fig.tight_layout()
    fig.savefig(output / 'learning_curves.png', dpi=160)
    plt.close(fig)
    if split is None:
        return None
    pred = probabilities.argmax(axis=1)
    cm = confusion_matrix(split.y, pred, labels=np.arange(24))
    normalized = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    fig, axis = plt.subplots(figsize=(10, 8))
    heat = axis.imshow(normalized, cmap='Blues', vmin=0, vmax=1)
    axis.set(xticks=np.arange(24), yticks=np.arange(24), xticklabels=letters, yticklabels=letters,
             xlabel='Predicted letter', ylabel='True letter', title='Test confusion matrix (row-normalized)')
    fig.colorbar(heat, ax=axis, label='Fraction within true class')
    fig.tight_layout()
    fig.savefig(output / 'confusion_matrix.png', dpi=160)
    plt.close(fig)
    wrong = np.flatnonzero(pred != split.y)[:16]
    fig, axes = plt.subplots(4, 4, figsize=(9, 9))
    for index, axis in enumerate(axes.flat):
        axis.axis('off')
        if index < len(wrong):
            row = wrong[index]
            image = split.x[row].reshape(28, 28)
            if normalization == 'center':
                image = image + .5
            elif normalization == 'image-center':
                # 此模式已移除各图均值，无法恢复绝对亮度；按图缩放便于查看形状。
                image = (image - image.min()) / max(float(np.ptp(image)), 1e-8)
            axis.imshow(image, cmap='gray', vmin=0, vmax=1)
            axis.set_title(f'True {letters[split.y[row]]} / pred {letters[pred[row]]}\np={probabilities[row, pred[row]]:.2f}', fontsize=10)
    fig.suptitle('First 16 test errors in official row order (not selected for severity)')
    fig.tight_layout()
    fig.savefig(output / 'error_examples.png', dpi=160)
    plt.close(fig)
    with (output / 'confusion_matrix.csv').open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['true/predicted', *letters])
        writer.writerows([letter, *row] for letter, row in zip(letters, cm))
    return cm


def check_unfinished_tasks(args):
    """只检查此次运行需要的空；在创建输出目录之前给出可读提示。"""
    root = Path(__file__).resolve().parent
    required = [('network.py', None)]
    if args.augment == 'mild':
        required.append(('augmentation.py', {'shift_batch', 'augment_batch'}))
    elif args.stress_validation:
        required.append(('augmentation.py', {'shift_batch'}))
    unfinished = []
    for filename, functions in required:
        tree = ast.parse((root / filename).read_text(encoding='utf-8'))
        for function in ast.walk(tree):
            if not isinstance(function, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            if functions is not None and function.name not in functions:
                continue
            for node in ast.walk(function):
                if not isinstance(node, ast.Raise):
                    continue
                error = node.exc
                name = error.func if isinstance(error, ast.Call) else error
                if isinstance(name, ast.Name) and name.id == 'NotImplementedError':
                    unfinished.append(f'{filename}:{node.lineno}（{function.name}）')
    if unfinished:
        raise SystemExit('请先完成本次运行所需的填空，并删除对应的 raise NotImplementedError：\n'
                         + '\n'.join(unfinished)
                         + '\n基础实验无需完成增强附加题，请保持 --augment none。')


def train(args):
    start = time.perf_counter()
    check_unfinished_tasks(args)
    args.data_dir = str(resolve_data_dir(args.data_dir))
    # 默认每次运行使用独立目录，便于学生反复调试；显式指定路径时仍保护已有结果。
    if args.output is None:
        tag = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        args.output = str(Path(__file__).resolve().parent / 'output' / ('run_' + tag))
    output = Path(args.output).expanduser().resolve()
    args.output = str(output)
    output.mkdir(parents=True, exist_ok=True)
    if (output / 'config.json').exists():
        raise FileExistsError(f'{output}: refusing to overwrite an existing experiment')
    config = vars(args).copy()
    config['started_utc'] = datetime.now(timezone.utc).isoformat()
    config['framework'] = 'NumPy, explicit backpropagation, mini-batch SGD'
    config['objective'] = 'mean cross entropy + l2/2 * sum(W**2); biases unregularized'
    config['checkpoint_rule'] = args.selection
    config['runtime'] = {'python': platform.python_version(), 'platform': platform.platform(),
                         'logical_cpu_count': os.cpu_count(),
                         'versions': {name: importlib.metadata.version(name) for name in
                                      ('numpy', 'matplotlib', 'pandas', 'scikit-learn', 'threadpoolctl')}}
    config['code_sha256'] = {name: sha256(Path(__file__).parent / name) for name in
                              ('main.py', 'network.py', 'sign_mnist_loader.py', 'augmentation.py')}
    write_json(output / 'config.json', config)
    print(f'结果保存目录：{output}', flush=True)
    splits, metadata = load_data(args.data_dir, args.validation_fraction, args.split_seed, args.normalization)
    write_json(output / 'data_metadata.json', metadata)
    np.savez_compressed(output / 'split_indices.npz', train=splits['train'].original_rows,
                        validation=splits['validation'].original_rows, test=splits['test'].original_rows)
    net = Network((784, *args.hidden_dims, 24), activation=args.activation, l2=args.l2, seed=args.seed)
    train_data, validation = splits['train'], splits['validation']
    print(json.dumps({'status': 'START', 'architecture': net.sizes, 'activation': args.activation,
                      'l2': args.l2, 'split_counts': metadata['split_counts'], 'threads': args.threads}), flush=True)
    rng = np.random.default_rng(args.seed + 1)
    augmentation_rng = np.random.default_rng(args.seed + 2)
    # 平移验证显式开启；基础实验不会调用尚未完成的增强选做题。
    stress_x = (shift_batch(validation.x, np.random.default_rng(104771), pixels=2)
                if args.stress_validation else None)

    def evaluate_validation_views(model):
        metrics = {'train': model.evaluate(train_data.x, train_data.y),
                   'validation': model.evaluate(validation.x, validation.y)}
        if stress_x is not None:
            metrics['stress_validation'] = model.evaluate(stress_x, validation.y)
        return metrics
    history, best_val, best_epoch, training_start = [], np.inf, None, time.perf_counter()
    initial = evaluate_validation_views(net)
    write_json(output / 'initial_metrics.json', initial)
    for epoch in range(1, args.epochs + 1):
        epoch_start = time.perf_counter()
        order = rng.permutation(len(train_data.y))
        lr = args.lr * args.lr_decay ** (epoch - 1)
        for offset in range(0, len(order), args.batch_size):
            index = order[offset:offset + args.batch_size]
            xb = train_data.x[index]
            if args.augment == 'mild':
                xb = augment_batch(xb, augmentation_rng)
            loss, dw, db = net.backward(xb, train_data.y[index])
            if not np.isfinite(loss) or not all(np.isfinite(grad).all() for grad in (*dw, *db)):
                raise FloatingPointError(f'nonfinite loss/gradient at epoch {epoch}; preserve this run and inspect learning rate')
            net.update(dw, db, lr)
        metrics = evaluate_validation_views(net)
        selection_loss = metrics['validation']['cross_entropy']
        if args.selection == 'stress-mean':
            selection_loss = (selection_loss + metrics['stress_validation']['cross_entropy']) / 2
        row = {'epoch': epoch, 'learning_rate': lr,
               **{name + '_' + key: value for name, scores in metrics.items() for key, value in scores.items()},
               'selection_loss': selection_loss, 'epoch_seconds': time.perf_counter() - epoch_start}
        history.append(row)
        if selection_loss < best_val:
            best_val, best_epoch = selection_loss, epoch
            net.save(output / 'best_model.npz')
        shifted_message = (f' shifted_acc={row["stress_validation_accuracy"]:.4f}'
                           if stress_x is not None else '')
        print(f'epoch={epoch:02d} lr={lr:.5f} train_ce={row["train_cross_entropy"]:.4f} '
              f'train_acc={row["train_accuracy"]:.4f} val_ce={row["validation_cross_entropy"]:.4f} '
              f'val_acc={row["validation_accuracy"]:.4f}{shifted_message} '
              f'seconds={row["epoch_seconds"]:.2f}', flush=True)
        write_json(output / 'progress.json', {'status': 'TRAINING', 'epoch': epoch, 'epochs': args.epochs,
                                             'best_epoch': best_epoch, **row})
        with (output / 'history.csv').open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(history[0]))
            writer.writeheader()
            writer.writerows(history)
    train_seconds = time.perf_counter() - training_start
    net.save(output / 'last_model.npz')
    # Only the validation-selected checkpoint is evaluated on the official test split.
    best = Network.load(output / 'best_model.npz')
    selected_metrics = {name: best.evaluate(split.x, split.y) for name, split in splits.items()
                        if name != 'test' or not args.validation_only}
    if stress_x is not None:
        selected_metrics['stress_validation'] = best.evaluate(stress_x, validation.y)
    if args.validation_only:
        plots(output, history, None, None, metadata['class_to_letter'], args.normalization)
        result = {'status': 'COMPLETE_VALIDATION_ONLY', 'architecture': list(best.sizes),
                  'activation': args.activation, 'l2': args.l2, 'epochs': args.epochs,
                  'normalization': args.normalization,
                  'best_epoch_by_validation_selection': best_epoch, 'selection_rule': args.selection,
                  'selected_validation_score': best_val,
                  'selected_checkpoint_metrics': selected_metrics, 'training_seconds': train_seconds,
                  'total_seconds': time.perf_counter() - start, 'threads': args.threads,
                  'test_evaluated': False, 'best_model_sha256': sha256(output / 'best_model.npz')}
        write_json(output / 'metrics.json', result)
        write_json(output / 'progress.json', result)
        print('COMPLETE_VALIDATION_ONLY', json.dumps(result), flush=True)
        return result
    test = splits['test']
    probabilities = best.predict_proba(test.x)
    predictions = probabilities.argmax(axis=1)
    assert float(np.mean(predictions == test.y)) == selected_metrics['test']['accuracy']
    np.savez_compressed(output / 'test_predictions.npz', original_rows=test.original_rows,
                        y_true=test.y, y_pred=predictions, probabilities=probabilities)
    report = classification_report(test.y, predictions, labels=np.arange(24),
                                    target_names=metadata['class_to_letter'], output_dict=True, zero_division=0)
    write_json(output / 'classification_report.json', report)
    cm = plots(output, history, test, probabilities, metadata['class_to_letter'], args.normalization)
    off_diagonal = cm.copy()
    np.fill_diagonal(off_diagonal, 0)
    top = np.argsort(off_diagonal.ravel())[::-1][:5]
    confusions = [{'true': metadata['class_to_letter'][int(i // 24)],
                   'predicted': metadata['class_to_letter'][int(i % 24)],
                   'count': int(off_diagonal.ravel()[i])} for i in top if off_diagonal.ravel()[i] > 0]
    parameter_count = sum(array.size for array in (*best.weights, *best.biases))
    result = {'status': 'COMPLETE', 'architecture': list(best.sizes), 'activation': args.activation,
              'l2': args.l2, 'parameters': parameter_count, 'epochs': args.epochs,
              'best_epoch_by_validation_selection': best_epoch, 'selection_rule': args.selection,
              'selected_validation_score': best_val, 'selected_checkpoint_metrics': selected_metrics,
              'test_macro_f1': float(f1_score(test.y, predictions, average='macro')),
              'test_samples': len(test.y), 'test_correct': int((predictions == test.y).sum()),
              'most_common_test_confusions': confusions, 'training_seconds': train_seconds,
              'total_seconds': time.perf_counter() - start, 'threads': args.threads,
              'test_used_for_training_or_checkpoint_selection': False,
              'test_scope': 'Official augmented static-letter image split; not a live-camera or unseen-signer evaluation.',
              'best_model_sha256': sha256(output / 'best_model.npz')}
    write_json(output / 'metrics.json', result)
    write_json(output / 'progress.json', {'status': 'COMPLETE', **result})
    print('COMPLETE', json.dumps(result), flush=True)
    return result


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', default=None,
                        help='默认读取脚本旁 data/ 中的 CSV；兼容 CSV 与脚本同层，可指定其他路径')
    parser.add_argument('--output', default=None,
                        help='默认在脚本旁 output/ 下自动创建独立运行目录；也可显式指定')
    parser.add_argument('--hidden-dims', type=int, nargs='+', default=[64])
    parser.add_argument('--activation', choices=['relu', 'sigmoid'], default='relu')
    parser.add_argument('--normalization', choices=['unit', 'center', 'image-center'], default='center')
    parser.add_argument('--augment', choices=['none', 'mild'], default='none',
                        help='mild 需完成 A1、A2；默认 none，只运行必做题')
    parser.add_argument('--stress-validation', action='store_true',
                        help='显式启用固定平移验证，需先完成 A1；不新增独立验证样本')
    parser.add_argument('--selection', choices=['validation', 'stress-mean'], default='validation')
    parser.add_argument('--epochs', type=int, default=80)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=.01)
    parser.add_argument('--lr-decay', type=float, default=1.)
    parser.add_argument('--l2', type=float, default=0.)
    parser.add_argument('--validation-fraction', type=float, default=.1)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--split-seed', type=int, default=42)
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--validation-only', action='store_true', help='Develop using train/validation curves without evaluating test metrics')
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.threads < 1 or args.lr <= 0 or args.lr_decay <= 0:
        parser.error('epochs, batch-size, threads, lr and lr-decay must be positive')
    if args.selection == 'stress-mean' and not args.stress_validation:
        parser.error('stress-mean requires --stress-validation (完成附加题 A1 后使用)')
    if args.augment == 'mild' and args.normalization != 'center':
        parser.error('mild augmentation requires center normalization')
    return args


if __name__ == '__main__':
    args = parse_args()
    with threadpool_limits(limits=args.threads):
        train(args)

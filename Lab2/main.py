"""Task2: Rice binary classification with a manual NumPy linear SVM.

Run python3 main.py --validation-only while developing; final evaluation omits
that flag. Each invocation receives an independent output directory.
"""
from __future__ import annotations
import argparse
import ast
import csv
from datetime import datetime
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import time
import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix, f1_score
from threadpoolctl import threadpool_limits
from rice_loader import load_data
from svm import LinearClassifier, ReferenceSVM
import visualize


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def evaluate(model, x, y, names=('Cammeo', 'Osmancik')):
    predicted = model.predict(x)
    return {'accuracy': float(accuracy_score(y, predicted)),
            'balanced_accuracy': float(balanced_accuracy_score(y, predicted)),
            'macro_f1': float(f1_score(y, predicted, average='macro')),
            'mean_data_loss': model.data_loss(x, y),
            'samples': len(y), 'correct': int(np.sum(predicted == y)),
            'confusion_matrix': confusion_matrix(y, predicted, labels=[-1, 1]).tolist(),
            'classification_report': classification_report(y, predicted, labels=[-1, 1],
                    target_names=list(names), output_dict=True, zero_division=0)}


def check_unfinished_tasks(args):
    tree = ast.parse((Path(__file__).parent / 'svm.py').read_text(encoding='utf-8'))
    required = {'decision_function', 'hinge_terms', 'objective', 'gradients', 'learning_rate', 'update'} if args.model == 'manual' else {'hinge_terms'}
    if args.loss == 'logistic' and args.model == 'manual':
        required.add('logistic_terms')
    if args.model == 'library-rbf':
        required.add('rbf_kernel')
    unfinished = []
    for n in ast.walk(tree):
        if isinstance(n, ast.FunctionDef) and n.name in required:
            if any(isinstance(x, ast.Raise) and isinstance(x.exc, ast.Call) and
                   isinstance(x.exc.func, ast.Name) and x.exc.func.id == 'NotImplementedError' for x in ast.walk(n)):
                unfinished.append(n.name)
    if unfinished:
        raise SystemExit('请先完成 svm.py 中的这些任务：' + ', '.join(unfinished))


def run(args, make_plots=True):
    check_unfinished_tasks(args)
    start = time.perf_counter()
    splits, meta = load_data(args.data_dir, args.scaling, args.split_seed)
    output = Path(args.output).expanduser().resolve() if args.output else (
        Path(__file__).parent / 'output' / ('run_' + datetime.now().strftime('%Y%m%d_%H%M%S_%f')))
    if (output / 'config.json').exists():
        raise FileExistsError(f'{output}: refusing to overwrite an existing run')
    output.mkdir(parents=True, exist_ok=True)
    config = vars(args).copy()
    config['runtime'] = {'python': platform.python_version(), 'platform': platform.platform(),
                         'versions': {n: importlib.metadata.version(n) for n in
                                      ['numpy', 'pandas', 'scipy', 'scikit-learn', 'matplotlib', 'threadpoolctl']}}
    config['code_sha256'] = {name: hashlib.sha256((Path(__file__).parent / name).read_bytes()).hexdigest()
                            for name in ['main.py', 'svm.py', 'rice_loader.py', 'visualize.py']}
    config['checkpoint_rule'] = 'minimum validation mean data loss, excluding regularization; earliest tie'
    write_json(output / 'config.json', config)
    write_json(output / 'data_metadata.json', meta)
    np.savez_compressed(output / 'split_indices.npz', **{k: s.original_rows for k, s in splits.items()})
    train, val = splits['train'], splits['validation']
    if args.model == 'manual':
        model = LinearClassifier(train.x.shape[1], args.regularization, args.lr, args.epochs,
                                 args.batch_size, args.loss, args.seed)
        model.fit(train.x, train.y, val.x, val.y, verbose=args.verbose)
        with (output / 'history.csv').open('w', newline='', encoding='utf-8') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(model.history[0]))
            writer.writeheader()
            writer.writerows(model.history)
        if make_plots:
            visualize.learning_curves(model.history, output / 'learning_curves.png', args.loss)
    else:
        kernel = 'linear' if args.model == 'library-linear' else 'rbf'
        c = args.C if args.C is not None else (1 / (len(train.y) * args.regularization) if kernel == 'linear' else 1.0)
        gamma = args.gamma if args.gamma is not None else 1 / train.x.shape[1]
        model = ReferenceSVM(kernel, C=c, gamma=gamma).fit(train.x, train.y)
        write_json(output / 'reference_parameters.json', {'C': c, 'gamma': gamma, 'kernel': kernel,
                   'n_support': model.n_support, 'loss': 'hinge'})
    model.save(output / 'best_model.npz')
    metrics = {k: evaluate(model, s.x, s.y) for k, s in splits.items()
               if k != 'test' or not args.validation_only}
    result = {'status': 'COMPLETE_VALIDATION_ONLY' if args.validation_only else 'COMPLETE',
              'model': args.model, 'loss': args.loss if args.model == 'manual' else 'hinge',
              'scaling': args.scaling, 'regularization': args.regularization,
              'best_epoch': model.best_epoch,
              'training_seconds': model.training_seconds, 'total_seconds': time.perf_counter() - start,
              'selected_checkpoint_metrics': metrics, 'test_evaluated': not args.validation_only,
              'test_used_for_training_or_selection': False}
    if not args.validation_only:
        test = splits['test']
        np.savez_compressed(output / 'test_predictions.npz', original_rows=test.original_rows,
                            y_true=test.y, y_pred=model.predict(test.x), scores=model.decision_function(test.x))
        if make_plots:
            visualize.confusion_matrix(np.asarray(metrics['test']['confusion_matrix']), output / 'confusion_matrix.png')
    write_json(output / 'metrics.json', result)
    print(json.dumps({'output': str(output), 'status': result['status'], 'best_epoch': model.best_epoch,
                      'validation_accuracy': metrics['validation']['accuracy'],
                      'training_seconds': model.training_seconds}), flush=True)
    return model, result, splits, output


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-dir', default=None)
    p.add_argument('--output', default=None)
    p.add_argument('--model', choices=['manual', 'library-linear', 'library-rbf'], default='manual')
    p.add_argument('--loss', choices=['hinge', 'logistic'], default='hinge')
    p.add_argument('--scaling', choices=['standard', 'none'], default='standard')
    p.add_argument('--regularization', type=float, default=0.01)
    p.add_argument('--lr', type=float, default=0.05)
    p.add_argument('--epochs', type=int, default=150)
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--split-seed', type=int, default=42)
    p.add_argument('--threads', type=int, default=2)
    p.add_argument('--C', type=float, default=None)
    p.add_argument('--gamma', type=float, default=None)
    p.add_argument('--validation-only', action='store_true')
    p.add_argument('--verbose', action='store_true')
    a = p.parse_args(argv)
    if a.regularization <= 0 or a.lr <= 0 or a.epochs < 1 or a.batch_size < 1 or a.threads < 1:
        p.error('regularization, lr, epochs, batch-size and threads must be positive')
    if (a.C is not None and a.C <= 0) or (a.gamma is not None and a.gamma <= 0):
        p.error('C and gamma must be positive')
    if a.model != 'manual' and a.loss != 'hinge':
        p.error('library SVM references use hinge loss')
    return a


if __name__ == '__main__':
    args = parse_args()
    with threadpool_limits(limits=args.threads):
        run(args)

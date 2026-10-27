"""Supplied 2-D geometry exercise. Add --rbf after completing optional A2.

Defaults to validation-only evaluation. --final-test opens the fixed test split.
"""
import argparse
from datetime import datetime
from pathlib import Path
import numpy as np
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from threadpoolctl import threadpool_limits
from main import evaluate, write_json
from svm import ReferenceSVM
import visualize


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--rbf', action='store_true')
    p.add_argument('--C', type=float, default=1.0)
    p.add_argument('--gamma', type=float, default=1.0)
    p.add_argument('--final-test', action='store_true')
    p.add_argument('--output', default=None)
    a = p.parse_args(argv)
    if a.C <= 0 or a.gamma <= 0:
        p.error('C and gamma must be positive')
    x, y = make_moons(n_samples=600, noise=.2, random_state=42)
    y = 2 * y - 1
    idx = np.arange(len(y))
    dev, te = train_test_split(idx, test_size=.2, random_state=42, stratify=y)
    tr, va = train_test_split(dev, test_size=.25, random_state=43, stratify=y[dev])
    mean, std = x[tr].mean(axis=0), x[tr].std(axis=0)
    x = (x - mean) / std
    out = Path(a.output).resolve() if a.output else Path(__file__).parent / 'output' / ('geometry_' + datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    if (out / 'geometry_results.json').exists():
        raise FileExistsError('Existing geometry run is protected; choose a new output directory')
    models = {'Linear SVM': ReferenceSVM('linear', a.C, a.gamma).fit(x[tr], y[tr])}
    if a.rbf:
        try:
            models['RBF SVM'] = ReferenceSVM('rbf', a.C, a.gamma).fit(x[tr], y[tr])
        except NotImplementedError:
            raise SystemExit('请先完成 svm.py 中 OPTIONAL A2，再使用 --rbf')
    out.mkdir(parents=True, exist_ok=True)
    result = {'data': 'Fixed make_moons, n=600, noise=0.2, random_state=42',
              'split_counts': {'train': len(tr), 'validation': len(va), 'test': len(te)},
              'C': a.C, 'gamma': a.gamma, 'standardizer_fit_on': 'training rows only',
              'test_evaluated': a.final_test,
              'models': {k: {'validation': evaluate(m, x[va], y[va], names=('Moon 0', 'Moon 1')),
                          **({'test': evaluate(m, x[te], y[te], names=('Moon 0', 'Moon 1'))} if a.final_test else {})} for k, m in models.items()}}
    write_json(out / 'geometry_results.json', result)
    visualize.toy_boundaries(x, y, tr, te if a.final_test else va, models, out / 'boundaries.png')
    print(out)
    return result


if __name__ == '__main__':
    with threadpool_limits(limits=2):
        main()

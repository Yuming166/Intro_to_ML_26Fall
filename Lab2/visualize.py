"""Plotting is supplied; students focus on the classifier and interpretation."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def learning_curves(history, path, title='Linear classifier'):
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    epoch = [r['epoch'] for r in history]
    for name, label in [('train', 'Train'), ('validation', 'Validation')]:
        axes[0].plot(epoch, [r[name + '_data_loss'] for r in history], label=label)
        axes[1].plot(epoch, [r[name + '_accuracy'] for r in history], label=label)
    axes[0].plot(epoch, [r['train_objective'] for r in history], '--', label='Train objective (incl. L2)', alpha=.8)
    axes[0].set(xlabel='Epoch', ylabel='Mean data loss', title=title)
    axes[1].set(xlabel='Epoch', ylabel='Accuracy', ylim=(0, 1.02))
    for ax in axes:
        ax.grid(alpha=.25)
        ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def confusion_matrix(cm, path, names=('Cammeo', 'Osmancik')):
    fig, ax = plt.subplots(figsize=(4.6, 4))
    im = ax.imshow(cm, cmap='Blues')
    ax.set(xticks=range(2), yticks=range(2), xticklabels=names, yticklabels=names,
           xlabel='Predicted', ylabel='True', title='Fixed test split')
    for (i, j), value in np.ndenumerate(cm):
        ax.text(j, i, str(value), ha='center', va='center', color='white' if value > np.max(cm) / 2 else 'black')
    fig.colorbar(im, ax=ax, shrink=.8)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def toy_boundaries(x, y, train_ids, test_ids, models, path):
    fig, axes = plt.subplots(1, len(models), figsize=(5 * len(models), 4.1))
    axes = np.atleast_1d(axes)
    a = np.linspace(x[:, 0].min() - .5, x[:, 0].max() + .5, 140)
    b = np.linspace(x[:, 1].min() - .5, x[:, 1].max() + .5, 140)
    xx, yy = np.meshgrid(a, b)
    grid = np.column_stack([xx.ravel(), yy.ravel()])
    for ax, (name, model) in zip(axes, models.items()):
        values = model.decision_function(grid).reshape(xx.shape)
        ax.contourf(xx, yy, values >= 0, levels=[-.5, .5, 1.5], colors=['#DDEBF7', '#FCE4D6'], alpha=.8)
        ax.contour(xx, yy, values, levels=[-1, 0, 1], colors=['#555555'] * 3,
                   linestyles=['--', '-', '--'], linewidths=[.8, 1.5, .8])
        ax.scatter(x[train_ids, 0], x[train_ids, 1], c=y[train_ids], cmap='coolwarm', s=12, alpha=.45)
        ax.scatter(x[test_ids, 0], x[test_ids, 1], c=y[test_ids], cmap='coolwarm', s=27,
                   edgecolors='black', linewidths=.6)
        if hasattr(model, 'support_x'):
            ax.scatter(model.support_x[:, 0], model.support_x[:, 1], s=50,
                       facecolors='none', edgecolors='gray', linewidths=.7)
        ax.set(title=name, xlabel='Standardized feature 1', ylabel='Standardized feature 2')
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)

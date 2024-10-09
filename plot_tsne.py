import numpy as np


import os
import numpy as np
from sklearn.metrics import pairwise_distances
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from matplotlib.colors import Normalize
from openTSNE import TSNE

np.random.seed(20)
plt.rcParams["figure.figsize"] = (7, 4.8)
import matplotlib
import pickle

matplotlib.rcParams.update({"font.size": 20})

from scripts.kernels import get_local_kernel
import seaborn as sns
import argparse as ap
import pandas as pd

pt = {"C": 6, "N": 7, "O": 8, "S": 16, "F": 9, "H": 1}
pt1 = {6:'C', 7:"N", 8:"O", 16:"S", 9:"F", 1:"H"}


def plot_tsne(
    x,
    y,
    target_name,
    training_name,
    selected_atom,
    ax=None,
    title=None,
    draw_legend=True,
    draw_centers=False,
    draw_cluster_labels=False,
    colors=None,
    legend_kwargs=None,
    label_order=None,
    **kwargs,
):

    if ax is None:
        _, ax = matplotlib.pyplot.subplots(figsize=(7, 8))

    if title is not None:
        ax.set_title(title)

    plot_params = {"alpha": kwargs.get("alpha", 0.6), "s": kwargs.get("s", 1)}

    # Set up sizes and alphas via trick
    s = np.array((y + 1) * (y + 18), dtype=int)
    alphas = np.clip((y + 1) * 0.4, a_min=0, a_max=1)

    alphas = np.zeros_like(y, dtype=float)
    alphas[np.where(y==0)] = 1.0
    alphas[np.where(y==1)] = 1.0 # {6: 0.125 / 2, 16: 20/845}[selected_atom]
    alphas[np.where(y==2)] = 1.0

    s[np.where(y==2)] = s[0] * 2

    classes = np.unique(y)
    default_colors = [i['color'] for i,j in zip(matplotlib.rcParams["axes.prop_cycle"](), range(10))]
    colors = {0: default_colors[7], 1: default_colors[8], 2: 'red'}

    point_colors = np.array(list(map(colors.get, y)))


    target_mask = (y==2)

    ax.scatter(
        x[~target_mask, 0],
        x[~target_mask, 1],
        c=point_colors[~target_mask],
        s=s[~target_mask],
        alpha=alphas[~target_mask],
        rasterized=True,
    )  # , **plot_params)


    ax.scatter(
        x[target_mask, 0],
        x[target_mask, 1],
        c=point_colors[target_mask],
        s=s[target_mask],
        alpha=alphas[target_mask],
        #edgecolors='black',
        rasterized=True,
        marker='x',
    )  # , **plot_params)

    # Hide ticks and axis
    ax.set_xticks([]), ax.set_yticks([]), ax.axis("off")

    if draw_legend:
        legend_handles = [
            matplotlib.lines.Line2D(
                [],
                [],
                marker="s",
                color="w",
                markerfacecolor=colors[yi],
                ms=10,
                alpha=1,
                linewidth=0,
                label=yi,
                markeredgecolor="k",
            )
            for yi in classes[::-1]
        ]
        legend_kwargs_ = dict(loc="upper center",
                              prop={'size': 12},
                              handletextpad=0, columnspacing=1,
                              bbox_to_anchor=(0.5, 0), frameon=False, ncol=len(classes))
        if legend_kwargs is not None:
            legend_kwargs_.update(legend_kwargs)
        ax.legend(handles=legend_handles, labels=['target','selected from QM7','other in QM7'], **legend_kwargs_)

    plt.tight_layout()
    plt.savefig(f"interpret_figs/tsne_{target_name}_{training_name}.pdf")
    # plt.show()
    return




target_name = 'penicillin'

training_set_names = [
    "h_algo_0_reps",
    "h_algo_1_reps",
    "h_random_reps",
    "h_cur_reps",
    "h_sml_reps",
    "h_fps_reps",
]

alg_name = {
    "h_algo_0_reps":'ILP(p=0)',
    "h_algo_1_reps":'ILP(p=1)',
    "h_random_reps":'random',
    "h_cur_reps":'CUR',
    "h_sml_reps":'SML',
    "h_fps_reps":'FPS',
        }

for training_name in training_set_names:

    data = np.load(f'interpret_figs/tsne_{target_name}_{training_name}.npz')

    x, y = data['x'], data['y']

    selected_atom = 6

    perplexity = {6: 250,
                  16: 4,
                  8: 80,  # old
                  7: 90,  # old
                  }

    plot_tsne(x, y, target_name,
        f"{training_name}_{selected_atom}_local_perp{perplexity[selected_atom]}",
        selected_atom,
        title = f'{alg_name[training_name]}',
        #title = f'{target_name} ({pt1[selected_atom]}) – {alg_name[training_name]}',
    )

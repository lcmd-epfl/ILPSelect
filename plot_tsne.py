import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

np.random.seed(20)
plt.rcParams["figure.figsize"] = (7, 4.8)
matplotlib.rcParams.update({"font.size": 20})

pt = {"C": 6, "N": 7, "O": 8, "S": 16, "F": 9, "H": 1}
pt1 = {6:'C', 7:"N", 8:"O", 16:"S", 9:"F", 1:"H"}


def plot_tsne(x, y, output_path=None, selected_atom=None, title=None):

    _, ax = matplotlib.pyplot.subplots(figsize=(7, 8))

    ax.set_title(title)

    # Set up sizes and alphas via trick
    s = np.array((y + 1) * (y + 18), dtype=int)
    alphas = np.clip((y + 1) * 0.4, a_min=0, a_max=1)

    alphas = np.zeros_like(y, dtype=float)
    alphas[np.where(y==0)] = 1.0
    alphas[np.where(y==1)] = {6: 0.125 / 2, 16: 20/845, 8: 20/80, 7:20/80}[selected_atom]
    alphas[np.where(y==2)] = 1.0

    s[np.where(y==2)] = s[0] * 3

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
    )


    ax.scatter(
        x[target_mask, 0],
        x[target_mask, 1],
        c=point_colors[target_mask],
        s=s[target_mask],
        alpha=alphas[target_mask],
        #edgecolors='black',
        rasterized=True,
        marker='x',
    )

    # Hide ticks and axis
    ax.set_xticks([]), ax.set_yticks([]), ax.axis("off")

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
    legend_kwargs = dict(loc="upper center",
                         prop={'size': 12},
                         handletextpad=0, columnspacing=1,
                         bbox_to_anchor=(0.5, 0), frameon=False, ncol=len(classes))
    ax.legend(handles=legend_handles, labels=['target','selected from QM7','other in QM7'], **legend_kwargs)

    plt.tight_layout()
    plt.savefig(output_path)
    return


selected_atom = 8
target_name = 'penicillin'

keys = ["algo_0", "algo_1", "random", "cur", "sml", "fps"]

alg_name = {
        "algo_0":'ILP(p=0)',
        "algo_1":'ILP(p=1)',
        "random":'random',
        "cur":'CUR',
        "sml":'SML',
        "fps":'FPS',
        }

perplexity = {6: 500,
              16: 4,
              8: 80,
              7: 90,
              }

for training_name in training_set_names:

    data = np.load(f'interpret_figs/tsne_{target_name}_{training_name}.npz')
    x, y = data['x'], data['y']

    plot_tsne(x, y, selected_atom=selected_atom,
        title = f'{alg_name[training_name]}',
        #title = f'{target_name} ({pt1[selected_atom]}) – {alg_name[training_name]}',
        output_path=f"interpret_figs/tsne_{target_name}_{training_name}.pdf",
    )

import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt



def plot_tsne(fig, ax, x, y, selected_atom=None, title=None, rasterized=True):

    ax.set_title(title)

    s = np.full(len(y), fill_value={8: 18, 7:18, 6:9, 16:18}[selected_atom])

    alphas = np.clip((y + 1) * 0.4, a_min=0, a_max=1)

    alphas = np.zeros_like(y, dtype=float)
    alphas[np.where(y==0)] = 1.0
    alphas[np.where(y==1)] = 1.0 #{6: 0.125 / 2, 16: 20/845, 8: 20/80, 7:20/80}[selected_atom]
    alphas[np.where(y==2)] = 1.0

    s[np.where(y==2)] = 90
    s[np.where(y==1)] = s[0]

    classes = np.unique(y)
    #default_colors = [i['color'] for i,j in zip(matplotlib.rcParams["axes.prop_cycle"](), range(10))]
    #colors = {0: default_colors[0], 1: default_colors[1], 2: default_colors[8]}
    #colors = {0: default_colors[0], 1: default_colors[1], 2: default_colors[7]}
    #colors = {0: default_colors[0], 1: default_colors[1], 2: default_colors[2]}
    #colors = {0: default_colors[0], 1: default_colors[1], 2: 'white'}
    #colors = {0: default_colors[0], 1: default_colors[1], 2: 'black'}
    #colors = {0: default_colors[0], 1: default_colors[1], 2: default_colors[6]}
    #colors = {0: default_colors[7], 1: default_colors[8], 2: 'red'}
    #colors = {0: default_colors[0], 1: default_colors[1], 2: 'white'}
    #colors = {0: default_colors[2], 1: default_colors[3], 2: default_colors[0]}
    #colors = {0: 'cyan', 1: 'magenta', 2: 'yellow'}
    colors = {0: '#0000FF', 1: '#00FF00', 2: '#FF0000'}

    point_colors = np.array(list(map(colors.get, y)))


    target_mask = (y==2)

    ax.scatter(
        x[~target_mask, 0],
        x[~target_mask, 1],
        c=point_colors[~target_mask],
        s=s[~target_mask],
        alpha=alphas[~target_mask],
        rasterized=rasterized,
    )

    ax.scatter(
        x[target_mask, 0],
        x[target_mask, 1],
        c=point_colors[target_mask],
        s=s[target_mask],
        alpha=alphas[target_mask],
        edgecolors='black',
        rasterized=rasterized,
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
    legend_kwargs = dict(loc="outside lower center",
                         prop={'size': 24},
                         handletextpad=0, columnspacing=1,
                         bbox_to_anchor=(0.5, 0), frameon=False, ncol=len(classes))

    fig.legend(handles=legend_handles, labels=['target','selected from QM7','other in QM7'], **legend_kwargs)

    return



selected_atom = 8
selected_atom = 6
target_name = 'penicillin'
rasterized = True

algos = ["algo_0", "algo_1", "fps", "cur", "sml", "random"]

alg_name = {
        "algo_0":'ILP(p=0)',
        "algo_1":'ILP(p=1)',
        "random":'random',
        "cur":'CUR',
        "sml":'SML',
        "fps":'FPS',
        }

perplexity = {6: 500, 16: 4, 8: 80, 7: 90}

np.random.seed(20)
matplotlib.rcParams.update({"font.size": 32})
pt1 = {6:'C', 7:"N", 8:"O", 16:"S", 9:"F", 1:"H"}

fig, axs = matplotlib.pyplot.subplots(2, 3, figsize=(8*3, 8*2))
for i, algo in enumerate(algos):

    data = np.load(f"interpret_figs/tsne/tsne_{target_name}_{selected_atom}_perp{perplexity[selected_atom]}_{algo}.npz")
    x, y = data['x'], data['y']

    ax = axs[i//3,i%3]

    plot_tsne(fig, ax, x, y, selected_atom=selected_atom,
        title = f'{alg_name[algo]}',
        #title = f'{target_name} ({pt1[selected_atom]}) – {alg_name[algo]}',
        rasterized=False,
    )

output_path=f"interpret_figs/tsne/tsne_{target_name}_{selected_atom}_perp{perplexity[selected_atom]}.png"
fig.tight_layout()
fig.savefig(output_path, dpi=600)

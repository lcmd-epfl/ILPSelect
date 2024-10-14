import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt



def plot_tsne(fig, ax, x, y, selected_atom=None, title=None, rasterized=True):

    ax.set_title(title, loc='left', fontsize=36)
    colors = {0: '#0000FF', 1: '#00FF00', 2: '#FF0000'}

    if x is not None:
        alphas = np.zeros_like(y, dtype=float)
        alphas[np.where(y==0)] = 1.0
        alphas[np.where(y==1)] = 1.0 #{6: 0.125 / 2, 16: 20/845, 8: 20/80, 7:20/80}[selected_atom]
        alphas[np.where(y==2)] = 1.0

        s = np.full(len(y), fill_value={8: 18, 7:18, 6:9, 16:36}[selected_atom])
        s[np.where(y==2)] = 90
        s[np.where(y==1)] = s[0]

        target_mask = (y==2)

        classes = np.unique(y)

        point_colors = np.array(list(map(colors.get, y)))

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
        for yi in range(2,-1,-1)
    ]
    legend_kwargs = dict(loc="outside lower center",
                         prop={'size': 32},
                         handletextpad=0, columnspacing=1,
                         bbox_to_anchor=(0.5, 0), frameon=False, ncol=3)

    fig.legend(handles=legend_handles, labels=['target','selected from QM7','other in QM7'], **legend_kwargs)

    return




algos = ["algo_0", "algo_1", "fps", "cur", "sml", "random"]

alg_name = {
        "algo_0":'ILP(p=0)',
        "algo_1":'ILP(p=1)',
        "random":'random',
        "cur":'CUR',
        "sml":'SML',
        "fps":'FPS',
        }

pt = {6:'C', 7:"N", 8:"O", 16:"S", 9:"F", 1:"H"}
perplexity = {6: 500, 16: 4, 8: 80, 7: 90}

#'sitagliptin' 'raltegravir'
for target in ['penicillin', 'apixaban', 'imatinib', 'oseltamivir', 'oxycodone',
               'pemetrexed', 'pregabalin', 'salbutamol', 'sildenafil', 'troglitazone']:
    print(target)
    for selected_atom in [6, 7, 8, 16]:
        print(selected_atom)

        fig, axs = matplotlib.pyplot.subplots(2, 3, figsize=(8*3, 8*2))
        for i, (label, algo) in enumerate(zip('abcdefgh', algos)):
            try:
                data = np.load(f"interpret_figs/tsne/tsne_{target}_{selected_atom}_perp{perplexity[selected_atom]}_{algo}.npz")
            except:
                data = {'x': None, 'y': None}
            plot_tsne(fig, axs[i//3,i%3], data['x'], data['y'],
                      selected_atom=selected_atom,
                      title = f'({label}) {alg_name[algo]}',
                      rasterized=True,
            )

        fig.tight_layout()
        if target=='penicillin' and selected_atom==6:
            dpi = 600
            plt.subplots_adjust(bottom=0.05, right=1, left=0)
        else:
            dpi = 300
            plt.subplots_adjust(bottom=0.05, right=1, top=0.88, left=0)
            fig.suptitle(f'{target} ({pt[selected_atom]})', fontsize=48)
        output_path=f"interpret_figs/tsne/tsne_{target}_{selected_atom}_perp{perplexity[selected_atom]}.pdf"
        fig.savefig(output_path, dpi=dpi)
        plt.close()

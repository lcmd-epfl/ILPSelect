import matplotlib.pyplot as plt
import numpy as np
import argparse as ap
import pandas as pd
plt.rcParams["figure.figsize"] = (6.4,6.4)
import matplotlib
matplotlib.rcParams.update({'font.size': 18})

Ha2kcal = 627.51

def parse_args():
    parser = ap.ArgumentParser()
    parser.add_argument('-t', '--target', default='all') # can be all
    parser.add_argument('-d', '--database', default='drugs') #drugs, qm7, qm9, qm7qm9
    parser.add_argument('-p', '--property', default='') # energy, dipole, gap
    args = parser.parse_args()
    return args

def get_lc(target, method, pen=0, database='drugs', property='energy'):

    if method == 'algo':
        lc = np.load(f'learning_curves/{property}/{method}_FCHL_qm7_{target}_{pen}.npz')
    else:
        lc = np.load(f'learning_curves/{property}/{method}_FCHL_qm7_{target}.npz')
    if property == 'dipole' or property == 'gapeV':
        if method == 'random':
            return lc['train_sizes'], lc['all_maes_random']
        else:
            return lc['train_sizes'], lc['mae']

    if method == 'random':
        return lc['train_sizes'], lc['all_maes_random'] * Ha2kcal
    return lc['train_sizes'], lc['mae'] * Ha2kcal

def average_std(stds):
    return np.sqrt(np.sum(np.array(stds) ** 2, axis=0)) / len(stds)

def plot_single_target(args):
    # FOR NOW ONLY FOR DRUGS
    methods = ['algo', 'algo', 'fps', 'cur', 'sml', 'random']
    labels = ['ILP(p=0)', 'ILP(p=1)', 'FPS', 'CUR', 'SML', 'Random']
    colors = ['tab:blue', 'tab:blue', 'tab:green', 'tab:red', 'tab:orange', 'tab:purple']
    fig, ax = plt.subplots(nrows=1, ncols=1)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log", base=2)
    ax.set_xlabel("Training set size")
    ax.set_ylabel("$\hat{E}$ MAE [kcal/mol]")
    for i, method in enumerate(methods):
        if i == 0:
            tr_sizes_0, maes_0 = get_lc(args.target, method, pen=0)
            ax.plot(tr_sizes_0, maes_0, color=colors[i], linestyle='dashed', label=labels[i])
        elif i == 1:
            tr_sizes_1, maes_1 = get_lc(args.target, method, pen=1)
            ax.plot(tr_sizes_1, maes_1, color=colors[i], label=labels[i])
        elif method == 'random':
            tr_sizes, all_maes = get_lc(args.target, method)
            mean_maes, std_maes = np.mean(all_maes, axis=0), np.std(all_maes, axis=0)
            ax.errorbar(tr_sizes, mean_maes, yerr=std_maes, label=labels[i], color=colors[i])
        else:
            tr_sizes, maes = get_lc(args.target, method)
            ax.plot(tr_sizes, maes, label=labels[i], color=colors[i])

    ax.set_xticks([16, 32, 64, 128, 256, 512, 1024])
    ax.set_xticklabels(['16', '32', '64', '128', '256', '512', '1024'])


    if args.target == 'apixaban' or args.target == 'imatinib' or args.target == 'pemetrexed':
        ax.set_yticks([64, 128, 256])
        ax.set_yticklabels(['64', '128', '256'])

    elif args.target == 'oseltamivir':
        ax.set_yticks([8, 16, 32, 64, 128])
        ax.set_yticklabels(['8', '16', '32', '64', '128'])

    elif args.target == 'oxycodone':
        ax.set_yticks([64, 128])
        ax.set_yticklabels(['64', '128'])

    elif args.target == 'penicillin':
        ax.set_yticks([16, 32, 64, 128, 256])
        ax.set_yticklabels(['16', '32', '64', '128', '256'])

    plt.tight_layout()
    plt.legend()
    plt.savefig(f"plots/lcs_new/{args.target}.pdf")
    plt.show()

def plot_avg_targets(args, database='drugs', property='energy'):
    if database == 'drugs':
        targets = ['apixaban', 'imatinib', 'oseltamivir', 'oxycodone', 'pemetrexed', 'penicillin', 'pregabalin',
                   'salbutamol', 'sildenafil', 'troglitazone']
    elif database == 'qm7':
        targets = ['qm7_1251', 'qm7_3576', 'qm7_6163', 'qm7_1513', 'qm7_1246',
                   'qm7_2161', 'qm7_6118', 'qm7_5245', 'qm7_5107', 'qm7_3037']
    elif database == 'qm9':
        targets = ["121259",
                   "12351",
                   "35811",
                   "85759",
                   "96295",
                   "5696",
                   "31476",
                   "55607",
                   "68076",
                   "120425"]
    else:
        raise NotImplementedError('only qm7, qm9 and drugs are implemented')
    methods = ['algo', 'algo', 'fps', 'cur', 'sml', 'random']
    labels = ['ILP(p=0)', 'ILP(p=1)', 'FPS', 'CUR', 'SML', 'Random']
    mean_maes = {}
    mean_stds = [] # only for random
    for label in labels:
        mean_maes[label] = []

    for target in targets:
        for i, method in enumerate(methods):
            label = labels[i]
            if i == 0:
                tr_sizes, maes_0 = get_lc(target, method, pen=0, database=database, property=property)
                mean_maes[label].append(maes_0)
            elif i == 1:
                _, maes_1 = get_lc(target, method, pen=1, database=database, property=property)
                mean_maes[label].append(maes_1)
            elif method == 'random':
                tr_sizes, all_maes = get_lc(target, method, database=database, property=property)
                mean_maes_r, std_maes_r = np.mean(all_maes, axis=0), np.std(all_maes, axis=0)
                mean_maes[label].append(mean_maes_r)
                mean_stds.append(std_maes_r)
            else:
                tr_sizes, maes = get_lc(target, method, database=database, property=property)
                mean_maes[label].append(maes)

    colors = ['tab:blue', 'tab:blue', 'tab:green', 'tab:red', 'tab:orange', 'tab:purple']
    fig, ax = plt.subplots(nrows=1, ncols=1)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log", base=2)
    ax.set_xlabel("Training set size")

    if property in ['energy', ''] :
        lab = "$\hat{E}$"
        unit = 'kcal/mol'
    elif property == 'gapeV':
        lab = "$\Delta \epsilon$"
        unit = 'eV'
    elif property == 'dipole':
        lab = "$\mu$"
        unit = 'a.u.'
    ax.set_ylabel(f"Average {lab} MAE [{unit}]")

    for i, label in enumerate(labels):
        if i == 0:
            linestyle = 'dashed'
        else:
            linestyle = 'solid'
        if label != 'Random':
            ax.plot(tr_sizes, np.mean(mean_maes[label], axis=0), marker='*', label=label, color=colors[i], linestyle=linestyle)
        else:
            ax.errorbar(tr_sizes, np.mean(mean_maes[label], axis=0), average_std(mean_stds), marker='*', label=label, color=colors[i])

    if database == 'drugs':
        if property in ['energy', '']:
            yticks = [30, 40, 60, 90, 134, 200, 300]
            ax.set_yticks(yticks)
            ax.set_yticklabels([str(y) for y in yticks])
            ax.set_ylim(min(yticks), max(yticks))
        elif property == 'gapeV':
            ax.set_yticks([1, 2, 4, 8])
            ax.set_yticklabels(['1', '2', '4', '8'])
        elif property == 'dipole':
           ax.set_yticks([0.5, 1, 2, 4])
           ax.set_yticklabels(['0.5', '1', '2', '4'])


    elif database == 'qm7':
        if property in ['energy', '']:
            yticks = [0.5, 1, 2, 4, 8, 16, 32]
            ax.set_yticks(yticks)
            ax.set_yticklabels([str(y) for y in yticks])
            ax.set_ylim(min(yticks), max(yticks))
        elif property == 'dipole':
            ax.set_yticks([0.05, 0.12, 0.25, 0.5, 1])
            ax.set_yticklabels(['0.05', '0.12', '0.25', '0.5', '1'])
        elif property == 'gapeV':
            ax.set_yticks([0.12, 0.25, 0.5, 1, 2])
            ax.set_yticklabels(['0.12', '0.25', '0.5', '1', '2'])

    elif database == 'qm9':
        if property in ['energy', '']:
            yticks = [1, 2, 4, 8, 16, 32, 64]
            ax.set_yticks(yticks)
            ax.set_yticklabels([str(y) for y in yticks])
            ax.set_ylim(min(yticks), max(yticks))
        elif property == 'dipole':
            ax.set_yticks([0.25, 0.5, 1])
            ax.set_yticklabels(['0.25', '0.5', '1'])
        elif property == 'gapeV':
            ax.set_yticks([0.5, 1])
            ax.set_yticklabels(['0.5', '1'])

    ax.set_xticks([], minor=True)
    ax.set_xticks([16, 32, 64, 128, 256, 512, 1024])
    ax.set_xticklabels(['16', '32', '64', '128', '256', '512', '1024'])
    ax.set_xlim(12, 1100)
    plt.tight_layout()

    if database == 'drugs':
        plt.legend(fontsize='small')
    plt.savefig(f"plots/lcs_new/average_{database}_{property}.pdf")
    plt.show()



def plot_qm7qm9():

    #plt.rcParams["figure.figsize"] = (6,3)
    targets = {'qm7': ['qm7_1251', 'qm7_3576', 'qm7_6163', 'qm7_1513', 'qm7_1246',
                       'qm7_2161', 'qm7_6118', 'qm7_5245', 'qm7_5107', 'qm7_3037'],
               'qm9': ["121259", "12351", "35811", "85759", "96295",
                       "5696", "31476", "55607", "68076", "120425"]}

    titles = {'qm7': 'QM7', 'qm9':'QM9*'}
    yticks = {'qm7': [0.5, 1, 2, 4, 8, 16, 32], 'qm9': [1, 2, 4, 8, 16, 32, 64]}

    methods = ['ILP(p=0)', 'ILP(p=1)', 'FPS', 'CUR', 'SML', 'Random']

    keys = {
        "ILP(p=0)" : ('algo', 0),
        "ILP(p=1)" : ('algo', 1),
        "FPS"      : ('fps', None),
        "CUR"      : ('cur', None),
        "SML"      : ('sml', None),
        "Random"   : ('random', None),
    }

    linestyles = {
        "ILP(p=0)" : 'dashed',
        "ILP(p=1)" : 'solid',
        "FPS"      : 'solid',
        "CUR"      : 'solid',
        "SML"      : 'solid',
        "Random"   : 'solid',
    }

    colors = {
          "ILP(p=0)" :  'tab:blue',
          "ILP(p=1)" :  'tab:blue',
          "FPS"      :  'tab:green',
          "CUR"      :  'tab:red',
          "SML"      :  'tab:orange',
          "Random"   :  'tab:purple',
    }

    mean_stds = {}
    mean_maes = {}
    for database in targets:
        mean_stds[database] = [] # only for random
        mean_maes[database] = {label: [] for label in methods}
        for target in targets[database]:
            for method in methods:
                if method == 'Random':
                    tr_sizes, all_maes = get_lc(target, keys[method][0], database=database, property='')
                    mean_maes_r, std_maes_r = np.mean(all_maes, axis=0), np.std(all_maes, axis=0)
                    mean_maes[database][method].append(mean_maes_r)
                    mean_stds[database].append(std_maes_r)
                else:
                    tr_sizes, maes = get_lc(target, keys[method][0], pen=keys[method][1], database=database, property='')
                    mean_maes[database][method].append(maes)


    fig, axes = plt.subplots(1, 2, figsize=(11, 6), sharey=False)

    for i, database in enumerate(targets):
        axes[i].set_title(titles[database], fontweight='bold')
        for method in methods:
            label=method if i==0 else None
            if method=='Random':
                axes[i].errorbar(
                    tr_sizes, np.mean(mean_maes[database][method], axis=0), average_std(mean_stds[database]),
                    label=label, marker='*', color=colors[method], linestyle=linestyles[method]
                )
            else:
                axes[i].plot(
                     tr_sizes, np.mean(mean_maes[database][method], axis=0),
                     label=label, marker='*', color=colors[method], linestyle=linestyles[method]
                )

        axes[i].set_xscale('log', base=2)
        axes[i].set_xticks(tr_sizes)
        axes[i].get_xaxis().set_major_formatter(plt.ScalarFormatter())
        axes[i].set_xlabel('Training set size')
        axes[i].tick_params(axis='y', which='both')
        axes[i].set_yscale("log", base=2)
        axes[i].set_yticks(yticks[database])
        axes[i].set_yticklabels([str(y) for y in yticks[database]])
        axes[i].set_ylim(min(yticks[database]), max(yticks[database]))
        if i==0:
            axes[i].set_ylabel('Average $\hat{E}$ MAE [kcal/mol]')

    fig.legend(bbox_to_anchor=(1.1, 0.88))
    plt.savefig(f"plots/lcs_new/average_qm7qm9_energy.pdf")
    plt.show()


if __name__=='__main__':
    args = parse_args()
    if args.database=='qm7qm9':
        plot_qm7qm9()
    else:
        if not args.target == 'all':
            plot_single_target(args)
        else:
            plot_avg_targets(args, database=args.database, property=args.property)

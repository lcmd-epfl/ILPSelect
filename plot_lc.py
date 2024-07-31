import matplotlib.pyplot as plt
import numpy as np
import argparse as ap
import pandas as pd

Ha2kcal = 627.5

def parse_args():
    parser = ap.ArgumentParser()
    parser.add_argument('-t', '--target', default='sildenafil') # can be all
    args = parser.parse_args()
    return args

def get_lc(target, method, pen=0):
    if method == 'algo':
        lc = np.load(f'learning_curves/{method}_FCHL_qm7_{target}_{pen}.npz')
    else:
        lc = np.load(f'learning_curves/{method}_FCHL_qm7_{target}.npz')

    if method == 'random':
        return lc['train_sizes'], lc['all_maes_random'] * 627.5

    return lc['train_sizes'], lc['mae'] * 627.5

def average_std(stds):
    return np.sqrt(np.sum(np.array(stds) ** 2, axis=0)) / len(stds)

def plot_single_target(args):
    methods = ['algo', 'algo', 'fps', 'cur', 'sml', 'random']
    labels = ['ILP(p=0)', 'ILP(p=1)', 'FPS', 'CUR', 'SML', 'Random']
    colors = ['tab:blue', 'tab:blue', 'tab:green', 'tab:red', 'tab:orange', 'tab:purple']
    fig, ax = plt.subplots(nrows=1, ncols=1)
    ax.set_xscale("log")
    ax.set_yscale("log")
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

    plt.tight_layout()
    plt.legend()
    plt.savefig(f"plots/lcs_clean/{args.target}.pdf")
    plt.show()

def plot_avg_targets(args):
    targets = ['apixaban', 'imatinib', 'oseltamivir', 'oxycodone', 'pemetrexed', 'penicillin', 'pregabalin',
               'salbutamol', 'sildenafil', 'troglitazone']
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
                tr_sizes, maes_0 = get_lc(target, method, pen=0)
                mean_maes[label].append(maes_0)
            elif i == 1:
                _, maes_1 = get_lc(target, method, pen=1)
                mean_maes[label].append(maes_1)
            elif method == 'random':
                tr_sizes, all_maes = get_lc(target, method)
                mean_maes_r, std_maes_r = np.mean(all_maes, axis=0), np.std(all_maes, axis=0)
                mean_maes[label].append(mean_maes_r)
                mean_stds.append(std_maes_r)
            else:
                tr_sizes, maes = get_lc(target, method)
                mean_maes[label].append(maes)

    colors = ['tab:blue', 'tab:blue', 'tab:green', 'tab:red', 'tab:orange', 'tab:purple']
    fig, ax = plt.subplots(nrows=1, ncols=1)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Training set size")
    ax.set_ylabel("Average $\hat{E}$ MAE [kcal/mol]")

    for i, label in enumerate(labels):
        if i == 0:
            linestyle = 'dashed'
        else:
            linestyle = 'solid'
        if label != 'Random':
            ax.plot(tr_sizes, np.mean(mean_maes[label], axis=0), label=label, color=colors[i], linestyle=linestyle)
        else:
            ax.errorbar(tr_sizes, np.mean(mean_maes[label], axis=0), average_std(mean_stds), label=label, color=colors[i])

    plt.tight_layout()
    plt.legend()
    plt.savefig(f"plots/lcs_clean/average.pdf")
    plt.show()

args = parse_args()

if not args.target == 'all':
    plot_single_target(args)

else:
    plot_avg_targets(args)
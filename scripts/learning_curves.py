import os
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, train_test_split
import qmllib.kernels
from qmllib.kernels import get_local_kernel, get_local_symmetric_kernel
from qmllib.solvers import cho_solve


SIGMAS = [1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3, 1e4]
L2REGS = [1e-7, 1e-6, 1e-4]

KERNEL_CACHE = True


def get_kernel_cache(X, Q, repository_path, database, representation):
    K_full = {}
    for sigma in SIGMAS:
        fname = f'{repository_path}/data/kernel_{representation}_{database}_{sigma}.npy'
        if os.path.isfile(fname):
            K_full[sigma] = np.load(fname)
            print(f'using cached kernel from {fname}')
            assert len(K_full[sigma])==len(X)
        else:
            print(f'{sigma=}')
            K_full[sigma] = get_local_symmetric_kernel(X, Q, SIGMA=sigma)
            np.save(fname, K_full[sigma])
            print(f'caching kernel to {fname}')
    return K_full


def krr(kernel, properties, l2reg=1e-9):
    alpha = cho_solve(kernel, properties, l2reg=l2reg)
    return alpha


def train_model(X_train, atoms_train, y_train, sigma=1, l2reg=1e-9):
    K_train = get_local_symmetric_kernel(X_train, atoms_train, SIGMA=sigma)
    alpha_train = krr(K_train, y_train, l2reg=l2reg)
    return alpha_train


def train_predict_model(
    X_train, atoms_train, y_train, X_test, atoms_test, y_test, sigma=1, l2reg=1e-9
):
    alpha_train = train_model(X_train, atoms_train, y_train, sigma=sigma, l2reg=l2reg)

    K_test = get_local_kernel(X_train, X_test, atoms_train, atoms_test, SIGMA=sigma)
    y_pred = np.dot(K_test, alpha_train)
    mae = np.abs(y_pred - y_test)[0]
    return mae, y_pred


def opt_hypers(X_train, atoms_train, y_train):

    n_folds = 5
    kf = KFold(n_splits=n_folds)

    maes = np.zeros((len(SIGMAS), len(L2REGS)))

    for i, sigma in enumerate(SIGMAS):
        print(f'{sigma=}')
        K_train = get_local_symmetric_kernel(X_train, atoms_train, SIGMA=sigma)
        for j, l2reg in enumerate(L2REGS):
            print(f'{l2reg=}')
            fold_maes = []
            for train_index, val_index in kf.split(X_train):
                y_train_fold, y_val_fold = y_train[train_index], y_train[val_index]

                K_train_fold = np.copy(K_train[np.ix_(train_index, train_index)])  # just in case
                K_test_fold = K_train[np.ix_(val_index, train_index)]

                alpha_train = krr(K_train_fold, y_train_fold, l2reg=l2reg)
                y_pred = np.dot(K_test_fold, alpha_train)
                mae = np.abs(y_pred - y_val_fold)[0]
                fold_maes.append(mae)

            avg_mae = np.mean(fold_maes)
            maes[i, j] = avg_mae

    min_j, min_k = np.unravel_index(np.argmin(maes, axis=None), maes.shape)
    min_sigma = SIGMAS[min_j]
    min_l2reg = L2REGS[min_k]
    print(
        "min avg mae",
        maes[min_j, min_k],
        "for sigma=",
        min_sigma,
        "and l2reg=",
        min_l2reg,
    )

    return min_sigma, min_l2reg


def learning_curves(config, random=False):
    """
    Compute learning curves once for each prefix, and each target.

    Parameters:
        config: config dictionary. Must contain keys `"repository_folder"`, `"penalty_lc"`, `"representation"`, `"target_names"`,
            `"database"`, `"in_database"`, `"learning_curve_ticks"`, `"config_name"`, `"CV"`.
        random: set true for N-fold random learning curves
    """

    repository_path = config["repository_folder"]
    penalty = config["penalty_lc"]
    if not isinstance(penalty, list):
        penalty = [penalty]

    representation = config["representation"]
    targets = config["target_names"]
    database = config["database"]
    config_name=config["config_name"]
    learning_curve_ticks = config["learning_curve_ticks"]
    in_database = config["in_database"]
    if random:
        CV = config["CV"]
        curve_i = [('random', i) for i in range(CV)]
    else:
        curves = [e for e in config["learning_curves"] if e != "random"]
        for curve in curves:
            assert curve in [
                "algo",
                "sml",
                "fps",
                "cur",
                "full",
            ], "only algo, sml, fps and cur algorithms are handled"
        curve_i = []
        for curve in curves:
            if curve == "algo":
                for pen in penalty:
                    curve_i.append((curve, pen))
            else:
                curve_i.append((curve, None))

    DATA_PATH = f"{repository_path}data/{representation}_{database}_{config_name}.npz"
    database_info = np.load(DATA_PATH, allow_pickle=True)
    X = database_info["reps"]
    Q = database_info["ncharges"]

    if KERNEL_CACHE:
        K_full = get_kernel_cache(X, Q, repository_path, database, representation)

    frame = pd.read_csv(f"{repository_path}{database}/energies.csv")

    # y energies offset
    with open(f"{repository_path}data/atom_energy_coeffs.pickle", "rb") as f:
        atom_energy_coeffs = pickle.load(f)

    if "atomization energy / Ha" in frame.columns:
        y = frame["atomization energy / Ha"].values
    else:
        y = frame["energy / Ha"].values
        for i, mol_ncharges in enumerate(Q):
            for ncharge in mol_ncharges:
                y[i] -= atom_energy_coeffs[ncharge]

    assert (len(X) == len(y)) and (len(Q) == len(y)), "Mismatch between number of database representations, charges, and labels."

    for target_name in targets:
        TARGET_PATH = (
            f"{repository_path}data/{representation}_{target_name}.npz"
        )

        target_info = np.load(TARGET_PATH, allow_pickle=True)

        X_target = target_info["rep"]
        Q_target = target_info["ncharges"]

        if in_database:
            Y_PATH = f"{repository_path}{database}/energies.csv"
            y_target = (
                pd.read_csv(Y_PATH)
                .query("file == @target_name")["energy / Ha"]
                .iloc[0]
            )

        if not in_database:
            Y_PATH = f"{repository_path}targets/energies.csv"
            y_target = (
                pd.read_csv(Y_PATH)
                .query("file == @target_name+'.xyz'")["energy / Ha"]
                .iloc[0]
            )

        # y energies offset
        for ncharge in Q_target:
            y_target -= atom_energy_coeffs[ncharge]


        all_maes_random = []
        all_y_preds_random = []
        all_sigmas_random = []
        all_l2regs_random = []

        for curve, i in curve_i:

            if curve == "algo":
                RANKING_PATH = f"{repository_path}rankings/algo_{representation}_{database}_{target_name}_{i}.npy"
                SAVE_PATH = f"{repository_path}learning_curves/algo_{representation}_{database}_{target_name}_{pen}.npz"
            elif curve in ["random", "sml", "cur", "fps"]:
                ext = 'npz' if curve=='fps' else 'npy'
                if not in_database and curve in ["cur", "fps", "random"]:
                    RANKING_PATH = f"{repository_path}/rankings/{curve}_{representation}_{database}.{ext}"
                else:
                    RANKING_PATH = f"{repository_path}/rankings/{curve}_{representation}_{database}_{target_name}.{ext}"
                SAVE_PATH = f"{repository_path}learning_curves/{curve}_{representation}_{database}_{target_name}.npz"

            if curve == "full":
                opt_ranking = range(len(y))
                learning_curve_ticks = [len(y)]
            else:
                opt_ranking = np.load(RANKING_PATH, allow_pickle=True)

            if curve == 'fps':
                # FPS has a special structure of array of rankings for each tick
                opt_ranking = {len(opt_ranking[k]) : opt_ranking[k] for k in opt_ranking.keys()}
            elif curve == 'random':
                opt_ranking = opt_ranking[i]


            if in_database:
                assert target_name not in database_info['labels'][opt_ranking if curve!='fps' else opt_ranking[max(learning_curve_ticks)]]
                assert np.allclose(y[np.where(database_info['labels']==target_name)], y_target)

            maes = []
            y_preds = []
            sigmas = []
            l2regs = []

            for n in learning_curve_ticks:

                print(f'tick={n}')
                if curve == "fps":
                    ranking = opt_ranking[n]
                else:
                    ranking = opt_ranking[:n]

                min_sigma, min_l2reg = opt_hypers(X[ranking], Q[ranking], y[ranking])

                mae, y_pred = train_predict_model(
                    X[ranking],
                    Q[ranking],
                    y[ranking],
                    np.array([X_target]),
                    np.array([Q_target]),
                    y_target,
                    sigma=min_sigma,
                    l2reg=min_l2reg,
                )
                maes.append(mae)
                y_preds.append(y_pred)
                sigmas.append(min_sigma)
                l2regs.append(min_l2reg)

            if curve == 'random':
                all_maes_random.append(maes)
                all_y_preds_random.append(y_preds)
                all_sigmas_random.append(sigmas)
                all_l2regs_random.append(l2regs)
            else:
                np.savez(
                    SAVE_PATH,
                    train_sizes=learning_curve_ticks,
                    mae=np.array(maes),
                    y_pred=y_preds,
                    sigma=sigmas,
                    l2reg=l2regs,
                )
                print(f"Saved to file {SAVE_PATH}.")

        if len(all_maes_random)>0:
            np.savez(
                SAVE_PATH,
                train_sizes=learning_curve_ticks,
                all_maes_random=all_maes_random,
                all_y_preds = all_y_preds_random,
                all_sigmas = all_sigmas_random,
                all_l2regs = all_l2regs_random,
            )
            print(f"Saved to file {SAVE_PATH}.")

    return 0


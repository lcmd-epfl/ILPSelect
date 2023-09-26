import os
import pickle

import numpy as np
import pandas as pd
import qml
from qml.math import cho_solve
from sklearn.model_selection import KFold


def krr(kernel, properties, l2reg=1e-9):
    alpha = cho_solve(kernel, properties, l2reg=l2reg)
    return alpha


def get_kernel(X1, X2, charges1, charges2, sigma=1):
    K = qml.kernels.get_local_kernel(X1, X2, charges1, charges2, sigma)
    return K


def train_model(X_train, atoms_train, y_train, sigma=1, l2reg=1e-9):
    K_train = get_kernel(X_train, X_train, atoms_train, atoms_train, sigma=sigma)
    alpha_train = krr(K_train, y_train, l2reg=l2reg)
    return alpha_train


def train_predict_model(
    X_train, atoms_train, y_train, X_test, atoms_test, y_test, sigma=1, l2reg=1e-9
):
    alpha_train = train_model(X_train, atoms_train, y_train, sigma=sigma, l2reg=l2reg)

    K_test = get_kernel(X_train, X_test, atoms_train, atoms_test)
    y_pred = np.dot(K_test, alpha_train)
    mae = np.abs(y_pred - y_test)[0]
    return mae, y_pred


def opt_hypers(X_train, atoms_train, y_train):
    sigmas = [0.25, 0.5, 0.75, 1e0, 1.25, 1.5]
    l2regs = [1e-7, 1e-6, 1e-4]

    n_folds = 5
    kf = KFold(n_splits=n_folds)

    maes = np.zeros((len(sigmas), len(l2regs)))

    for i, sigma in enumerate(sigmas):
        for j, l2reg in enumerate(l2regs):
            fold_maes = []
            for train_index, val_index in kf.split(X_train):
                X_train_fold, X_val_fold = X_train[train_index], X_train[val_index]
                atoms_train_fold, atoms_val_fold = atoms_train[train_index], atoms_train[val_index]
                y_train_fold, y_val_fold = y_train[train_index], y_train[val_index]

                mae, _ = train_predict_model(
                    X_train_fold,
                    atoms_train_fold,
                    y_train_fold,
                    X_val_fold,
                    atoms_val_fold,
                    y_val_fold,
                    sigma=sigma,
                    l2reg=l2reg,
                )
                fold_maes.append(mae)

            avg_mae = np.mean(fold_maes)
            print("sigma", sigma, "l2reg", l2reg, "avg mae", avg_mae)
            maes[i, j] = avg_mae

    min_j, min_k = np.unravel_index(np.argmin(maes, axis=None), maes.shape)
    min_sigma = sigmas[min_j]
    min_l2reg = l2regs[min_k]
    print("min avg mae", maes[min_j, min_k], "for sigma=", min_sigma, "and l2reg=", min_l2reg)

    return min_sigma, min_l2reg


def learning_curves(repository_path, database, targets, representation, config, algorithms):
    """
    Compute learning curves once for each prefix, and each target. For N-fold random learning curves, use `learning_curves_random`.

    Parameters:
        parent_directory: absolute path to cluster/ folder
        database: name of database (str) eg "qm7"
        targets: array of target names (array(str))
        representation: name of representation (str) eg "FCHL"
        config: config dictionary. Must contain keys "penalty", "learning_curve_ticks"
        algorithms: list of algorithms to test (array(str)). WARNING only ["fragments", "sml"] is handled.
    """
    for algorithm in algorithms:
        assert algorithm in ["fragments", "sml"], "only fragments and sml algorithms are handled"

    pen = config["penalty"]
    DATA_PATH = f"{repository_path}cluster/data/{representation}_{database}.npz"
    database_info = np.load(DATA_PATH, allow_pickle=True)
    X = database_info["reps"]
    Q = database_info["ncharges"]

    y = pd.read_csv(f"{repository_path}{database}/energies.csv")["energy / Ha"].values

    # y energies offset
    with open(f"{repository_path}cluster/data/atom_energy_coeffs.pickle", "rb") as f:
        atom_energy_coeffs = pickle.load(f)

    for i, mol_ncharges in enumerate(Q):
        for ncharge in mol_ncharges:
            y[i] -= atom_energy_coeffs[ncharge]

    for algorithm in algorithms:
        for target_name in targets:
            TARGET_PATH = f"{repository_path}cluster/data/{representation}_{target_name}.npz"

            target_info = np.load(TARGET_PATH, allow_pickle=True)
            X_target = target_info["rep"]
            Q_target = target_info["ncharges"]

            if config["in_database"]:
                Y_PATH = f"{repository_path}{database}/energies.csv"
                y_target = (
                    pd.read_csv(Y_PATH).query("file == @target_name+'.xyz'")["energy / Ha"].iloc[0]
                )
            else:
                Y_PATH = f"{repository_path}cluster/targets/energies.csv"
                y_target = (
                    pd.read_csv(Y_PATH).query("file == @target_name+'.xyz'")["energies"].iloc[0]
                )

            # y energies offset
            for ncharge in Q_target:
                y_target -= atom_energy_coeffs[ncharge]

            # fragments algorithm ranking
            if algorithm == "fragments":
                RANKING_PATH = f"{repository_path}cluster/rankings/algo_{representation}_{database}_{target_name}_{pen}.npy"
            elif algorithm == "sml":
                RANKING_PATH = f"{repository_path}cluster/rankings/sml_{representation}_{database}_{target_name}.npy"
            elif algorithm in ["fps", "cur"]:
                if config["in_database"]:
                    RANKING_PATH = f"{repository_path}cluster/rankings/{algorithm}_{representation}_{database}_{target_name}.npy"
                else:
                    RANKING_PATH = f"{repository_path}cluster/rankings/{algorithm}_{representation}_{database}.npy"

            opt_ranking = np.load(RANKING_PATH)

            maes = []
            for n in config["learning_curve_ticks"]:
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

            maes = np.array(maes)

            if algorithm == "fragments":
                SAVE_PATH = f"{repository_path}cluster/learning_curves/algo_{representation}_{database}_{target_name}_{pen}.npz"
            else:
                SAVE_PATH = f"{repository_path}cluster/learning_curves/{algorithm}_{representation}_{database}_{target_name}.npz"

            np.savez(
                SAVE_PATH,
                train_sizes=config["learning_curve_ticks"],
                mae=maes,
                ranking_xyz=database_info["labels"][opt_ranking],
            )

            print(f"Saved to file {SAVE_PATH}.")

    return 0


def learning_curves_random(
    repository_path, database, targets, representation, config, CV, add_onto_old=True
):
    """
    Compute for CV-fold random learning curves.

    Parameters:
        repository_path:cluster/ absolute path to cluster/ folder
        database: name of database (str) eg "qm7"
        targets: array of target names (array(str))
        representation: name of representation (str) eg "FCHL"
        config: config dictionary. Must contain keys "learning_curve_ticks", "random_state"
        CV: number of iterations of random curve (int)
        add_onto_old: if some random curves already exist, we will append onto them (bool)
    """

    if config["random_state"] != None:
        print("WARNING: random_state is fixed -- all random subsets are identical!")

    DATA_PATH = f"{repository_path}cluster/data/{representation}_{database}.npz"
    database_info = np.load(DATA_PATH, allow_pickle=True)
    X = database_info["reps"]
    Q = database_info["ncharges"]
    database_labels = database_info["labels"]

    y = pd.read_csv(f"{repository_path}{database}/energies.csv")["energy / Ha"].values

    # y energies offset
    with open(f"{repository_path}cluster/data/atom_energy_coeffs.pickle", "rb") as f:
        atom_energy_coeffs = pickle.load(f)

    for i, mol_ncharges in enumerate(Q):
        for ncharge in mol_ncharges:
            y[i] -= atom_energy_coeffs[ncharge]

    for target_name in targets:
        TARGET_PATH = f"{repository_path}cluster/data/{representation}_{target_name}.npz"

        target_info = np.load(TARGET_PATH, allow_pickle=True)
        X_target = target_info["rep"]
        Q_target = target_info["ncharges"]

        if config["in_database"]:
            # label of target
            Y_PATH = f"{repository_path}{database}/energies.csv"
            y_target = (
                pd.read_csv(Y_PATH).query("file == @target_name+'.xyz'")["energy / Ha"].iloc[0]
            )

            # removing target from database
            mask = database_labels != target_name
            X = X[mask]
            Q = Q[mask]
            database_labels = database_labels[mask]

        else:
            Y_PATH = f"{repository_path}cluster/targets/energies.csv"
            y_target = pd.read_csv(Y_PATH).query("file == @target_name+'.xyz'")["energies"].iloc[0]

        # y energies offset
        for ncharge in Q_target:
            y_target -= atom_energy_coeffs[ncharge]

        SAVE_PATH = f"{repository_path}cluster/learning_curves/random_{representation}_{database}_{target_name}.npz"

        all_maes_random = []
        opt_rankings = []

        if add_onto_old and os.path.isfile(SAVE_PATH):
            old_random = np.load(SAVE_PATH, allow_pickle=True)
            all_maes_random = old_random["all_maes_random"].tolist()
            opt_rankings = old_random["ranking_xyz"].tolist()

        # five fold cross validation
        CV = 5
        kf = KFold(n_splits=CV, shuffle=True, random_state=130)

        for i, (train_index, test_index) in enumerate(kf.split(X)):
            maes_random = []

            for n in config["learning_curve_ticks"]:
                min_sigma, min_l2reg = opt_hypers(
                    X[train_index][:n], Q[train_index][:n], y[train_index][:n]
                )
                print(min_sigma, min_l2reg)

                mae, y_pred = train_predict_model(
                    X[train_index][:n],
                    Q[train_index][:n],
                    y[train_index][:n],
                    np.array([X_target]),
                    np.array([Q_target]),
                    y_target,
                    sigma=min_sigma,
                    l2reg=min_l2reg,
                )

                maes_random.append(mae)
                print("Random", n, mae)

            all_maes_random.append(maes_random)
            opt_rankings.append(database_labels[train_index])

        # all_maes_random = np.array(all_maes_random)

        np.savez(
            SAVE_PATH,
            train_sizes=config["learning_curve_ticks"],
            all_maes_random=all_maes_random,
            ranking_xyz=opt_rankings,
        )

        print(f"Saved to file {SAVE_PATH}.")

    return 0

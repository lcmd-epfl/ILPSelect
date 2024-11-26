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
        else:
            print(f'{sigma=}')
            K_full[sigma] = get_local_symmetric_kernel(X, Q, SIGMA=sigma)
            np.save(fname, K_full[sigma])
            print(f'caching kernel to {fname}')
    return K_full


def krr(kernel, properties, l2reg=1e-9):
    alpha = cho_solve(kernel, properties, l2reg=l2reg)
    return alpha


def get_kernel(X1, X2, charges1, charges2, sigma=1):
    K = get_local_kernel(X1, X2, charges1, charges2, sigma)
    return K


def train_model(X_train, atoms_train, y_train, sigma=1, l2reg=1e-9):
    K_train = get_kernel(X_train, X_train, atoms_train, atoms_train, sigma=sigma)
    alpha_train = krr(K_train, y_train, l2reg=l2reg)
    return alpha_train


def train_predict_model(
    X_train, atoms_train, y_train, X_test, atoms_test, y_test, sigma=1, l2reg=1e-9
):
    alpha_train = train_model(X_train, atoms_train, y_train, sigma=sigma, l2reg=l2reg)

    K_test = get_kernel(X_train, X_test, atoms_train, atoms_test, sigma=sigma)
    y_pred = np.dot(K_test, alpha_train)
    mae = np.abs(y_pred - y_test)[0]
    return mae, y_pred


def opt_hypers(X_train, atoms_train, y_train):

    n_folds = 5
    kf = KFold(n_splits=n_folds)

    maes = np.zeros((len(SIGMAS), len(L2REGS)))

    for i, sigma in enumerate(SIGMAS):
        print(f'{sigma=}')
        K_train = get_kernel(X_train, X_train, atoms_train, atoms_train, sigma=sigma)
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


def learning_curves(config):
    """
    Compute learning curves once for each prefix, and each target. For N-fold random learning curves, use `learning_curves_random`.

    Parameters:
        config: config dictionary. Must contain keys `"repository_folder"`, `"penalty_lc"`, `"representation"`, `"target_names"`,
            `"database"`, `"in_database"`, `"learning_curve_ticks"`, `"config_name"`.
    """

    repository_path = config["repository_folder"]
    penalty = config["penalty_lc"]
    if not isinstance(penalty, list):
        penalty = [penalty]
    representation = config["representation"]
    targets = config["target_names"]
    database = config["database"]
    curves = [e for e in config["learning_curves"] if e != "random"]
    config_name=config["config_name"]
    learning_curve_ticks = config["learning_curve_ticks"]
    in_database = config["in_database"]

    for curve in curves:
        assert curve in [
            "algo",
            "sml",
            "fps",
            "cur",
            "full",
        ], "only algo, sml, fps and cur algorithms are handled"

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

    for curve in curves:
        for pen in (penalty if curve == "algo" else [None]):
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
                else:
                    Y_PATH = f"{repository_path}targets/energies.csv"
                    y_target = (
                        pd.read_csv(Y_PATH)
                        .query("file == @target_name+'.xyz'")["energy / Ha"]
                        .iloc[0]
                    )

                # y energies offset
                for ncharge in Q_target:
                    y_target -= atom_energy_coeffs[ncharge]

                # algo curve ranking
                if curve == "algo":
                    RANKING_PATH = f"{repository_path}rankings/algo_{representation}_{database}_{target_name}_{pen}.npy"
                elif curve == "sml":
                    RANKING_PATH = f"{repository_path}rankings/sml_{representation}_{database}_{target_name}.npy"
                elif curve == "cur":
                    if in_database:
                        RANKING_PATH = f"{repository_path}rankings/{curve}_{representation}_{database}_{target_name}.npy"
                    else:
                        RANKING_PATH = f"{repository_path}rankings/{curve}_{representation}_{database}.npy"
                elif curve == "fps":
                    if in_database:
                        RANKING_PATH = f"{repository_path}rankings/{curve}_{representation}_{database}_{target_name}.npz"
                    else:
                        RANKING_PATH = f"{repository_path}rankings/{curve}_{representation}_{database}.npz"

                if curve == "full":
                    opt_ranking = range(len(y))
                    learning_curve_ticks = [len(y)]
                else:
                    opt_ranking = np.load(RANKING_PATH, allow_pickle=True)

                maes = []
                y_preds = []
                sigmas = []
                l2regs = []
                i=0
                for n in learning_curve_ticks:
                    print(f'tick={n}')

                    # FPS has a special structure of array of rankings for each tick
                    # throws an error if there are more learning curve ticks than entries in the ranking
                    if curve == "fps":
                        ranking = opt_ranking["arr_"+str(i)]
                        i+=1
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

                maes = np.array(maes)

                if curve == "algo":
                    SAVE_PATH = f"{repository_path}learning_curves/algo_{representation}_{database}_{target_name}_{pen}.npz"
                else:
                    SAVE_PATH = f"{repository_path}learning_curves/{curve}_{representation}_{database}_{target_name}.npz"

                np.savez(
                    SAVE_PATH,
                    train_sizes=learning_curve_ticks,
                    mae=maes,
                    y_pred=y_preds,
                    sigma=sigmas,
                    l2reg=l2regs,
                    # ranking_xyz=database_info["labels"][opt_ranking],
                )

                print(f"Saved to file {SAVE_PATH}.")

    return 0


def learning_curves_random(config, add_onto_old=True):
    """
    Compute for CV-fold random learning curves.

    Parameters:
        config: config dictionary. Must contain keys `"repository_folder"`, `"representation"`, `"target_names"`,
            `"database"`, `"in_database"`, `"learning_curve_ticks"`, `"config_name"`, `"CV"`, `"random_state"`.
        add_onto_old: if some random curves already exist, we will append onto them (bool)
    """

    repository_path = config["repository_folder"]
    representation = config["representation"]
    targets = config["target_names"]
    database = config["database"]
    CV = config["CV"]
    config_name=config["config_name"]
    in_database=config["in_database"]
    learning_curve_ticks=config["learning_curve_ticks"]

    if config["random_state"] != None:
        print("WARNING: random_state is fixed -- all random subsets are identical!")

    DATA_PATH = f"{repository_path}data/{representation}_{database}_{config_name}.npz"
    database_info = np.load(DATA_PATH, allow_pickle=True)


    X = database_info["reps"]
    Q = database_info["ncharges"]
    database_labels = database_info["labels"]

    database_energies = pd.read_csv(f"{repository_path}{database}/energies.csv")

    # y energies offset
    with open(f"{repository_path}data/atom_energy_coeffs.pickle", "rb") as f:
        atom_energy_coeffs = pickle.load(f)

    if "atomization energy / Ha" in database_energies.columns:
        y = database_energies["atomization energy / Ha"].values
    else:
        y = database_energies["energy / Ha"].values
        for i, mol_ncharges in enumerate(Q):
            for ncharge in mol_ncharges:
                y[i] -= atom_energy_coeffs[ncharge]

    # shuffle
    N = len(X)
    inds = np.arange(N)
    np.random.shuffle(inds)
    X=X[inds]
    Q=Q[inds]
    database_labels=database_labels[inds]
    y=y[inds]

    for target_name in targets:
        TARGET_PATH = (
            f"{repository_path}data/{representation}_{target_name}.npz"
        )

        target_info = np.load(TARGET_PATH, allow_pickle=True)

        X_target = target_info["rep"]
        Q_target = target_info["ncharges"]


        # y_target definition
        if in_database:
            # label of target
            if "atomization energy / Ha" in database_energies.columns:
                y_target = database_energies.query("file == @target_name")[
                    "atomization energy / Ha"
                ].iloc[0]
            else:
                y_target = database_energies.query("file == @target_name")[
                    "energy / Ha"
                ].iloc[0]
                # y energies offset
                for ncharge in Q_target:
                    y_target -= atom_energy_coeffs[ncharge]
            # removing target from database
            mask = (database_labels != target_name)
            X = X[mask]
            Q = Q[mask]
            print(y_target)
            y_target = y[np.logical_not(mask)][0]
            print(y_target)
            database_labels = database_labels[mask]
            y=y[mask]

        else:
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
        ranking = []
        all_y_preds = []
        all_sigmas = []
        all_l2regs = []

        if add_onto_old and os.path.isfile(SAVE_PATH):
            old_random = np.load(SAVE_PATH, allow_pickle=True)
            all_maes_random = old_random["all_maes_random"].tolist()
            ranking = old_random["ranking_xyz"].tolist()
            all_y_preds = old_random['all_y_preds'].tolist()
            all_sigmas  = old_random['all_sigmas'].tolist()
            all_l2regs  = old_random['all_l2regs'].tolist()

        # five fold cross validation
        for i in range(CV):
            # we don't use the test indices since we test on the target (label y_target)
            X_train, _, Q_train, _, database_labels_train, _, y_train, _ = train_test_split(X, Q, database_labels, y, test_size=.2, random_state=config["random_state"])
            maes_random = []
            y_preds = []
            sigmas = []
            l2regs = []
            for n in learning_curve_ticks:

                min_sigma, min_l2reg = opt_hypers(
                    X_train[:n], Q_train[:n], y_train[:n]
                )

                mae, y_pred = train_predict_model(
                    X_train[:n],
                    Q_train[:n],
                    y_train[:n],
                    np.array([X_target]),
                    np.array([Q_target]),
                    y_target,
                    sigma=min_sigma,
                    l2reg=min_l2reg,
                )

                maes_random.append(mae)
                y_preds.append(y_pred)
                sigmas.append(min_sigma)
                l2regs.append(min_l2reg)
                print("Random", n, mae)

            all_maes_random.append(maes_random)
            ranking.append(database_labels_train)
            all_y_preds.append(y_preds)
            all_sigmas.append(sigmas)
            all_l2regs.append(l2regs)
        print("All MAEs random", all_maes_random)

        SAVE_PATH = f"{repository_path}learning_curves/random_{representation}_{database}_{target_name}.npz"

        np.savez(
            SAVE_PATH,
            train_sizes=learning_curve_ticks,
            all_maes_random=all_maes_random,
            ranking_xyz=ranking,
            all_y_preds = all_y_preds,
            all_sigmas = all_sigmas,
            all_l2regs = all_l2regs,
        )

        print(f"Saved to file {SAVE_PATH}.")

    return 0

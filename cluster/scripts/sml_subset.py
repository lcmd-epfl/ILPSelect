import numpy as np
import qml
from qml.math import cho_solve


def krr(kernel, properties, l2reg=1e-9):
    alpha = cho_solve(kernel, properties, l2reg=l2reg)
    return alpha


def get_kernel(X1, X2, charges1, charges2, sigma=1):
    K = qml.kernels.get_local_kernel(X1, X2, charges1, charges2, sigma)
    return K


def get_ranking(X, X_target, Q, Q_target):
    K = get_kernel(X, X_target, Q, Q_target, sigma=1)
    return np.argsort(K)[::-1]


def sml_subset(parent_folder, database, targets, representation, N, remove_target_from_database):
    """
    Generate SML subsets of size N for each target from the database.

    Parameters:
        parent_folder: absolute path of folder containing data/ folder with needed representations
        database: name of database (str) eg. "qm7"
        targets: array of names (array(str))
        representation: name of rep (str) eg. "FCHL"
        N: size of each subset
    """
    DATA_PATH = f"{parent_folder}data/"
    database_info = np.load(f"{DATA_PATH}{representation}_{database}.npz", allow_pickle=True)

    database_reps = database_info["reps"]
    database_ncharges = database_info["ncharges"]

    for target_name in targets:
        target_info = np.load(f"{DATA_PATH}{representation}_{target_name}.npz", allow_pickle=True)

        ranking = get_ranking(
            database_reps,
            np.array([target_info["rep"]]),
            database_ncharges,
            np.array([target_info["ncharges"]]),
        )[: N + 1]

        if remove_target_from_database:
            # the best ranking fragment should be itself!
            assert database_info["labels"][ranking[0]] == target_name
            ranking = ranking[:1]
        else:
            # discard the n+1st, not that it matters.
            ranking = ranking[:-1]

        SAVE_PATH = f"{parent_folder}rankings/sml_{representation}_{database}_{target_name}.npy"

        np.save(SAVE_PATH, ranking)

        print(
            f"Saved ranking of {N} closest fragments of database {database} to target {target_name} to {SAVE_PATH}."
        )

    return 0

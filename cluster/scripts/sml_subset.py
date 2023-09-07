import numpy as np
import qml

#old ranking with kernels
"""
def get_kernel(X1, X2, charges1, charges2, sigma=1):
    K = qml.kernels.get_local_kernel(X1, X2, charges1, charges2, sigma)
    return K


def get_ranking(X, X_target, Q, Q_target):
    K = get_kernel(X, X_target, Q, Q_target, sigma=1)
    return np.argsort(K[0])[::-1]
"""

#new ranking as closest point sampling 
def get_ranking(X_train, X_target):
    shape1, shape2 = X_train.shape[0], X_target.shape[0]
    distances = np.linalg.norm(X_train.reshape((shape1, -1)) - X_target.reshape((shape2, -1)), axis=1)

    # Sort the indices by the distances
    sorted_indices = np.argsort(distances)

    return sorted_indices

def sml_subset(parent_folder, database, targets, representation, N):
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

        mask = database_info["labels"] != target_name

        # old ranking with kernels
        """
        ranking = get_ranking(
            database_reps[mask],
            np.array([target_info["rep"]]),
            database_ncharges[mask],
            np.array([target_info["ncharges"]]),
        )[:N]
        """
        ranking = get_ranking(
            database_reps[mask],
            np.array([target_info["rep"]]),
        )[:N]

        SAVE_PATH = f"{parent_folder}rankings/sml_{representation}_{database}_{target_name}.npy"

        np.save(SAVE_PATH, ranking)

        print(
            f"Saved ranking of {N} closest fragments of database {database} to target {target_name} to {SAVE_PATH}."
        )

    return 0

import numpy as np
from skmatter.feature_selection import CUR, FPS


def cur_subset(config):
    """
    Generate CUR subsets of size N for each target from the database.

    Parameters:
        config: TODO
    """

    parent_folder = config["current_folder"]
    database = config["database"]
    targets = config["target_names"]
    representation = config["representation"]
    N = config["learning_curve_ticks"][-1]
    in_database = config["in_database"]

    DATABASE_PATH = f"{parent_folder}data/{representation}_{database}.npz"
    database_info = np.load(DATABASE_PATH, allow_pickle=True)

    database_reps = database_info["reps"]
    if representation == "FCHL":
        database_global_rep = np.array([np.sum(rep, axis=0) for rep in database_reps])
    else:
        print("Only FCHL is taken care of right now.")
        raise

    selector = CUR(n_to_select=N + 1)
    # transpose matrix to select samples instead of features
    selector.fit(database_global_rep.T)
    ranking = selector.selected_idx_

    SAVE_PATH = f"{parent_folder}rankings/cur_{representation}_{database}.npy"
    np.save(SAVE_PATH, ranking[:N])
    print(f"Saved CUR ranking of {N} fragments of database {database} to {SAVE_PATH}.")

    if not in_database:
        return 0

    for target_name in targets:
        target_index = np.where(database_info["labels"] == target_name)[0][0]
        ranking = ranking[ranking != target_index][:N]

        SAVE_PATH = f"{parent_folder}rankings/cur_{representation}_{database}_{target_name}.npy"
        np.save(SAVE_PATH, ranking)
        print(
            f"Saved CUR ranking of {N} fragments of database {database} without {target_name} to {SAVE_PATH}."
        )

    return 0


def fps_subset(config):
    """
    Generate FPS subsets of size N for each target from the database.

    Parameters:
        parent_folder: absolute path of folder containing data/ folder with needed representations
        database: name of database (str) eg. "qm7"
        targets: array of names (array(str))
        representation: name of rep (str) eg. "FCHL"
        N: size of each subset
        in_database: whether the targets are in the database and should be removed from the ranking or not (bool)
    """

    parent_folder = config["current_folder"]
    representation = config["representation"]
    targets = config["target_names"]
    database = config["database"]
    N = config["learning_curve_ticks"][-1]
    in_database = config["in_database"]

    DATABASE_PATH = f"{parent_folder}data/{representation}_{database}.npz"
    database_info = np.load(DATABASE_PATH, allow_pickle=True)

    database_reps = database_info["reps"]
    if representation == "FCHL":
        database_global_rep = np.array([np.sum(rep, axis=0) for rep in database_reps])
    else:
        print("Only FCHL is taken care of right now.")
        raise

    selector = FPS(n_to_select=N + 1)
    print(database_global_rep.shape)
    selector.fit(database_global_rep)
    ranking = selector.selected_idx_

    if not in_database:
        SAVE_PATH = f"{parent_folder}rankings/fps_{representation}_{database}.npy"
        np.save(SAVE_PATH, ranking[:N])
        print(f"Saved FPS ranking of {N} fragments of database {database} to {SAVE_PATH}.")
        return 0

    for target_name in targets:
        target_index = np.where(database_info["labels"] == target_name)[0][0]
        ranking = ranking[ranking != target_index][:N]

        SAVE_PATH = f"{parent_folder}rankings/fps_{representation}_{database}_{target_name}.npy"
        np.save(SAVE_PATH, ranking)
        print(
            f"Saved FPS ranking of {N} fragments of database {database} without {target_name} to {SAVE_PATH}."
        )

    return 0

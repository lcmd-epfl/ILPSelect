import numpy as np


# new ranking as closest point sampling
def get_ranking(X, X_target):
    database_global_rep = np.array([np.sum(rep, axis=0) for rep in X])
    target_global_rep = np.sum(X_target, axis=0)
    distances = np.linalg.norm(
        (database_global_rep - target_global_rep).astype(float), axis=1
    )
    # Sort the indices by the distances
    sorted_indices = np.argsort(distances)

    return sorted_indices


def sml_subset(config):
    """
    Generate SML subsets of size N for each target from the database.

    Parameters:
        config: TODO
    """

    parent_folder = config["repository_folder"]
    database = config["database"]
    targets = config["target_names"]
    representation = config["representation"]
    N = config["learning_curve_ticks"][-1]
    in_database = config["in_database"]
    config_name=config["config_name"]

    DATA_PATH = f"{parent_folder}data/"
    database_info = np.load(
        f"{DATA_PATH}{representation}_{database}_{config_name}.npz", allow_pickle=True
    )

    database_reps = database_info["reps"]
    # database_ncharges = database_info["ncharges"]

    for target_name in targets:
        target_info = np.load(
            f"{DATA_PATH}{representation}_{target_name}.npz", allow_pickle=True
        )
        target_rep = target_info["rep"]

        # closest point sampling
        if in_database:
            # i don't just remove the first element because for some reason sometimes it's not?
            ranking = get_ranking(database_reps, target_rep)[: N + 1]
            target_index = np.where(database_info["labels"] == target_name)[0][0]
            ranking = ranking[ranking != target_index][:N]
        else:
            ranking = get_ranking(database_reps, target_rep)[:N]

        SAVE_PATH = (
            f"{parent_folder}rankings/sml_{representation}_{database}_{target_name}.npy"
        )

        np.save(SAVE_PATH, ranking)

        print(
            f"Saved ranking of {N} closest fragments of database {database} to target {target_name} to {SAVE_PATH}."
        )

    return 0

import numpy as np
import pandas as pd


def random_subset(repository_folder, database, N, random_state=None, target_to_remove=None):
    """
    Randomly shuffle database and return the first N indices.

    Parameters:
        database_path: path to database with `energies.csv` frame present containing column "file"
        N: number of indices to return
    """
    database_files = pd.read_csv(f"{repository_folder}{database}/energies.csv").query(
        "file != @target_to_remove"
    )["file"]

    ranking = database_files.sample(n=N, random_state=random_state).index

    # no need to save, it's kept in the learning_curve_random npz file.
    # SAVE_PATH = f"{repository_folder}cluster/rankings/random_{database}.npy"
    # np.save(SAVE_PATH, ranking)
    # print(f"Saved ranking of {N} random fragments of database {database} to {SAVE_PATH}.")

    return ranking

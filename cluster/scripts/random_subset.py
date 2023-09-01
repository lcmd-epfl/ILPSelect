import numpy as np
import pandas as pd


def random_subset(repository_folder, database, N, random_state=None):
    """
    Randomly shuffle database and return the first N indices.

    Parameters:
        database_path: path to database with `energies.csv` frame present containing column "file"
        N: number of indices to return
    """
    database_files = pd.read_csv(f"{repository_folder}{database}/energies.csv")["file"]

    SAVE_PATH = f"{repository_folder}cluster/rankings/random_{database}.npy"

    np.save(SAVE_PATH, database_files.sample(n=N, random_state=random_state).index)

    print(f"Saved ranking of {N} random fragments of database {database} to {SAVE_PATH}.")

    return 0

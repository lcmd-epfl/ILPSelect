import pdb
import pickle
import random

import numpy as np
import pandas as pd
import qml
from qml.math import cho_solve


def get_representations(mols, params, representation):
    # TODO: add other representations

    assert representation == "FCHL", "Only FCHL is implemented."

    max_natoms = max([len(mol.nuclear_charges) for mol in mols])
    elements = np.unique(np.concatenate([(mol.nuclear_charges) for mol in mols]))

    reps = np.array(
        [
            qml.representations.generate_fchl_acsf(
                mol.nuclear_charges,
                mol.coordinates,
                elements=elements,
                gradients=False,
                pad=max_natoms,
            )
            for mol in mols
        ]
    )
    nuclear_charges = np.array([mol.nuclear_charges for mol in mols], dtype=object)
    return reps, nuclear_charges


def generate_targets(targets, representation, parent_folder):
    """
    Generate representation of targets in parent_folder/data/.

    Parameters:
        targets: array of names (array(strings))
        representaion: representation name (string)
        parent_folder: above folder which contains a targets/ folder with the .xyz files
    """

    for target_name in targets:
        TARGET_PATH = f"{parent_folder}targets/{target_name}.xyz"

        target_mol = qml.Compound(TARGET_PATH)

        X_target, Q_target = get_representations(
            [target_mol], params=None, representation=representation
        )

        # to use in the fragments algo
        SAVE_PATH = f"{parent_folder}data/{representation}_{target_name}"
        np.savez(f"{SAVE_PATH}.npz", ncharges=Q_target[0], rep=X_target[0])

        print(f"Generated representation {representation} of target {target_name} in {SAVE_PATH}.")


def generate_database(database, representation, repository_folder):
    """
    Generate representation of full databases in repository_folder/cluster/data/ from xyz files in /repository_folder/{representation}
    There must be a `energies.csv` in the database folder with columns "file" and "energy / Ha".

    Parameters:
        database: name of database (str) eg "qm7"
        representation: name of representation (str) eg "FCHL"
        repository_folder: absolute path of rep folder
    """

    FRAGMENTS_PATH = f"{repository_folder}{database}/"

    fragments = pd.read_csv(f"{FRAGMENTS_PATH}energies.csv")

    file_names = fragments["file"].to_list()

    xyzs = fragments["file"].map(lambda x: x + ".xyz").to_list()

    mols = np.array([qml.Compound(f"{FRAGMENTS_PATH}{x}") for x in xyzs])

    X, Q = get_representations(mols, params=None, representation=representation)

    SAVE_PATH = f"{repository_folder}cluster/data/"

    np.savez(f"{SAVE_PATH}{representation}_{database}.npz", reps=X, labels=file_names, ncharges=Q)

    print(f"Generated representation {representation} of database {database} in {SAVE_PATH}.")

import numpy as np
import pandas as pd
import qml


def get_representations(mols, max_natoms=None, elements=None, representation="FCHL"):
    # TODO: add other representations

    assert representation == "FCHL", "Only FCHL is implemented."

    if max_natoms is None:
        print(
            "[WARNING] using max natoms of input molecules. This could create problems if the target is smaller than database."
        )
        max_natoms = max([len(mol.nuclear_charges) for mol in mols])
    if elements is None:
        print(
            "[WARNING] using ncharges of input molecules. This could create a segmentation fault if new ncharges are in the target."
        )
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
        ],
        # dtype=object,
    )
    nuclear_charges = np.array([mol.nuclear_charges for mol in mols], dtype=object)
    return reps, nuclear_charges


# to add attributes for get_representation()
class Mol(object):
    def __init__(self, nuclear_charges, coordinates):
        self.nuclear_charges = nuclear_charges
        self.coordinates = coordinates


def generate_targets(config):
    """
    Generate representation of targets in parent_folder/data/.
    The database must already be generated in order to keep the same parameters!

    Parameters:
        config: TODO
    """

    repository_folder = config["repository_folder"]
    database = config["database"]
    targets = config["target_names"]
    representation = config["representation"]

    DATA_PATH = f"{repository_folder}cluster/data/qm9_data.npz"
    database_info = np.load(DATA_PATH, allow_pickle=True)
    database_coordinates = database_info["coordinates"]
    database_nuclear_charges = database_info["charges"]

    # only used to not miss some new ncharges in targets, and natoms
    all_elements = np.unique(np.concatenate([(x) for x in database_nuclear_charges]))
    max_natoms = max([len(x) for x in database_nuclear_charges])

    for target_name in targets:
        # target_name is an index here
        target_mol = Mol(
            database_nuclear_charges[target_name], database_coordinates[target_name]
        )

        X_target, Q_target = get_representations(
            [target_mol],
            max_natoms=max_natoms,
            elements=all_elements,
            representation=representation,
        )

        # to use in the fragments algo
        SAVE_PATH = (
            f"{repository_folder}cluster/data/{representation}_{target_name}.npz"
        )
        np.savez(SAVE_PATH, ncharges=Q_target[0], rep=X_target[0])

        print(
            f"Generated representation {representation} of target {target_name} in {SAVE_PATH}."
        )

    return 0


def generate_database(config):
    """
    Generate representation of full databases in repository_folder/cluster/data/ from xyz files in /repository_folder/{database}
    There must be a `energies.csv` in the database folder with columns "file" and "energy / Ha".

    Parameters:
        config:TODO
    """

    repository_folder = config["repository_folder"]
    database = config["database"]
    targets = config["target_names"]
    representation = config["representation"]
    in_database = config["in_database"]

    DATA_PATH = f"{repository_folder}cluster/data/qm9_data.npz"
    database_info = np.load(DATA_PATH, allow_pickle=True)
    database_coordinates = database_info["coordinates"]
    database_nuclear_charges = database_info["charges"]

    assert len(database_coordinates) == len(database_nuclear_charges)

    mols = [
        Mol(database_nuclear_charges[i], database_coordinates[i])
        for i in range(len(database_coordinates))
    ]

    # only used to not miss some new ncharges in targets outside database
    target_mols = []

    all_elements = np.unique(
        np.concatenate(
            [(mol.nuclear_charges) for mol in mols]
            + [(mol.nuclear_charges) for mol in target_mols]
        )
    )

    max_natoms = np.max(
        [len(mol.nuclear_charges) for mol in mols]
        + [len(mol.nuclear_charges) for mol in target_mols]
    )

    X, Q = get_representations(
        mols,
        max_natoms=max_natoms,
        representation=representation,
        elements=all_elements,
    )

    SAVE_PATH = f"{repository_folder}cluster/data/{representation}_{database}.npz"

    np.savez(SAVE_PATH, reps=X, labels=database_info["index"], ncharges=Q)

    print(
        f"Generated representation {representation} of database {database} in {SAVE_PATH}."
    )

    return 0

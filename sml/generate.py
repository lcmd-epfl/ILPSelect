import qml
import numpy as np
from qml.math import cho_solve
import pandas as pd
import random
import pickle
import pdb
import matplotlib.pyplot as plt
random.seed(42)
np.random.seed(42)
#matplotlib font size 
plt.rcParams.update({'font.size': 18})

def get_representations(mols, params):
    max_natoms = max([len(mol.nuclear_charges) for mol in mols])
    elements = np.unique(np.concatenate([(mol.nuclear_charges) for mol in mols]))

    reps = np.array([qml.representations.generate_fchl_acsf(mol.nuclear_charges, 
                                                mol.coordinates,
                                                elements=elements,
                                                gradients=False,
                                                pad=max_natoms)
            for mol in mols])
    nuclear_charges = np.array([mol.nuclear_charges for mol in mols])
    return reps, nuclear_charges


if __name__ == '__main__':

    ALL_TARGETS = pd.read_csv("./targets/targets.csv")
    TARGETS_XYZ, TARGETS_y = ALL_TARGETS["xyz"].values, ALL_TARGETS["energies"].values



    #here everything in hartree units
    with open('atom_energy_coeffs.pickle', 'rb') as f:
        atom_energy_coeffs = pickle.load(f)
            

    FRAGMENTS_PATH = "/home/haeberle/molekuehl/qm7"
    FRAG_y = pd.read_csv(f"{FRAGMENTS_PATH}/energies_qm7.csv")
    xyzs, y_train =FRAG_y["file"].values, FRAG_y["energy / Ha"].values
    mols       = np.array([qml.Compound(f"{FRAGMENTS_PATH}/{x}.xyz") for x in xyzs])
    X, Q = get_representations(mols, params=None)
    # to use in the fragments algo
    np.savez(f"./results/data_qm7.npz", ncharges=Q, reps=X, labels=xyzs)


    for xyz_target, y_target in zip(TARGETS_XYZ, TARGETS_y):
        print("Target:", xyz_target)
        target_name = xyz_target.split(".")[0]
        TARGET_PATH = f"./targets/{xyz_target}"

        target_mol = qml.Compound(TARGET_PATH)

        X_target, Q_target = get_representations([target_mol], params=None)
        
        # to use in the fragments algo
        np.savez(f"./results/{target_name}.npz", ncharges=Q_target[0], rep=X_target[0])

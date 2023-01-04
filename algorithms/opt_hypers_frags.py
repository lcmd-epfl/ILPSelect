import qml
import numpy as np
from qml.math import cho_solve
from qml.kernels import get_local_kernel, get_local_kernels_gaussian
import pandas as pd
from glob import glob
import pickle
import matplotlib.pyplot as plt

def KRR(X_train, N_train, y_train, X_test, N_test, y_test, sigma, l2reg):
    assert len(X_train) == len(N_train)
    assert len(X_train) == len(y_train)
    assert len(X_test) == len(N_test)
    assert len(N_test) == len(y_test)

    X_train = np.concatenate(X_train, axis=0)
    K_train = get_local_kernels_gaussian(X_train, X_train, N_train, N_train,
                                         [sigma])[0]
    alpha = cho_solve(K_train, y_train, l2reg=l2reg)

    X_test = np.concatenate(X_test, axis=0)
    K_test = get_local_kernels_gaussian(X_test, X_train, N_test, N_train,
                                       [sigma])[0]
    y_pred = np.dot(K_test, alpha)
    return np.mean(np.abs(y_pred - y_test))


def hyperparam_opt(X_train, N_train, y_train, X_test, N_test, y_test):
    sigmas = [1, 10, 100, 1000]
    l2regs = [1e-10, 1e-7, 1e-4]
    maes = np.zeros((len(sigmas), len(l2regs)))

    for j, sigma in enumerate(sigmas):
        for k, l2reg in enumerate(l2regs):
            mae = KRR(X_train, N_train, y_train, X_test, N_test, y_test, sigma, l2reg)
            print('mae', mae, 'for sigma=', sigma, 'l2reg=', l2reg)
            maes[j,k] = mae
    min_j, min_k = np.unravel_index(np.argmin(maes, axis=None),
                                   maes.shape)
    min_sigma = sigmas[min_j]
    min_l2reg = l2regs[min_k]
    print('min mae ',maes[min_j, min_k],' for sigma=',min_sigma,
         ' and l2reg=',min_l2reg)
    return min_sigma, min_l2reg

if __name__ == "__main__":
    with open("atom_energy_coeffs.pickle", "rb") as f:
        atom_energy_coeffs = pickle.load(f)

    props = pd.read_csv("opt-amons-penicillin-target/energies.csv", names=['file', 'energy'])
    target_energy = float(props[props['file'] == 'penicillin.xyz']['energy'])
    target_mol = qml.Compound("opt-amons-penicillin-target/penicillin.xyz")
    for ncharge in target_mol.nuclear_charges:
        target_energy -= atom_energy_coeffs[ncharge]

    frag_indices = np.load('ordered_fragments_04-01-23.npy', allow_pickle=True)
    qm7_xyz = ['qm7/qm7_'+str(idx)+'.xyz' for idx in frag_indices]
    qm7_mols = [qml.Compound(x) for x in qm7_xyz]
    mbtypes = qml.representations.get_slatm_mbtypes([x.nuclear_charges for x in qm7_mols])
    target_mol_nuclear_charges = np.unique(target_mol.nuclear_charges)
    target_nat = len(target_mol.nuclear_charges)
    target_rep = np.array(qml.representations.generate_slatm(target_mol.coordinates,
                                                   target_mol.nuclear_charges, mbtypes=mbtypes, local=True))
    qm7_ncharges = np.array([x.nuclear_charges for x in qm7_mols])
    qm7_nat = np.array([len(x) for x in qm7_ncharges])
    qm7_reps = np.array([np.array(qml.representations.generate_slatm(x.coordinates, x.nuclear_charges, mbtypes=mbtypes,
                        local=True)) for x in qm7_mols])
    qm7_props = pd.read_csv("qm7/energies_qm7.csv", index_col=0)
    qm7_labels = [x.split('/')[-1].split('.xyz')[0] for x in qm7_xyz]
    qm7_energy = np.array([float(qm7_props[qm7_props['file'] == label]['energy / Ha']) for label in qm7_labels])
    for i, mol_ncharges in enumerate(qm7_ncharges):
        for ncharge in mol_ncharges:
            qm7_energy[i] -= atom_energy_coeffs[ncharge]

    sigma, l2reg = hyperparam_opt(qm7_reps, qm7_nat, qm7_energy, np.array([target_rep]), np.array([target_nat]),
                              np.array([target_energy]))


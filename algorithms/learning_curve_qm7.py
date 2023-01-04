import qml
import numpy as np
from qml.math import cho_solve
from qml.kernels import get_local_kernel, get_local_kernels_gaussian
import pandas as pd
import pickle
from glob import glob

with open('atom_energy_coeffs.pickle', 'rb') as f:
    atom_energy_coeffs = pickle.load(f)

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

def learning_curve(X, atoms, y, X_test, atoms_test, y_test, sigma=1, l2reg=1e-10, CV=3):
    train_fractions = np.logspace(-1, 0, num=10, endpoint=True)
    train_sizes = [int(len(y)*x) for x in train_fractions]

    maes = np.zeros((CV, 5))
    ints = np.arange(len(X))

    for i in range(CV):
        print("Shuffle training data iter...",i+1,"/",CV)
        np.random.seed(i)
        np.random.shuffle(ints)
        X = X[ints]
        atoms = atoms[ints]
        y = y[ints]

        for j, train_size in enumerate(train_sizes):
            print("train size", train_size)
            X_train = X[:train_size]
            y_train = y[:train_size]
            atoms_train = atoms[:train_size]

            mae = KRR(X_train, atoms_train, y_train,
                                        X_test, atoms_test, y_test, sigma=sigma,
                                        l2reg=l2reg)
            maes[i,j] = mae

    mean_maes = np.mean(maes, axis=0)
    stdev = np.std(maes, axis=0)

    return train_sizes, mean_maes, stdev


if __name__ == "__main__":
    props = pd.read_csv("opt-amons-penicillin-target/energies.csv", names=['file', 'energy'])
    target_energy = float(props[props['file'] == 'penicillin.xyz']['energy'])
    target_mol = qml.Compound("opt-amons-penicillin-target/penicillin.xyz")
    for ncharge in target_mol.nuclear_charges:
        target_energy -= atom_energy_coeffs[ncharge]

    qm7_xyz = glob('qm7/*.xyz')
    qm7_mols = [qml.Compound(x) for x in qm7_xyz]
    qm7_ncharges = np.array([x.nuclear_charges for x in qm7_mols])
    mbtypes = qml.representations.get_slatm_mbtypes(qm7_ncharges)
    target_rep = qml.representations.generate_slatm(target_mol.coordinates,
                                                   target_mol.nuclear_charges, mbtypes=mbtypes, local=True) 
    target_N = len(target_mol.nuclear_charges)
    qm7_N = np.array([len(x) for x in qm7_ncharges])
    qm7_reps = np.array([qml.representations.generate_slatm(x.coordinates, x.nuclear_charges,
                                                            mbtypes=mbtypes, local=True)
                        for x in qm7_mols])
    qm7_props = pd.read_csv("qm7/energies_qm7.csv", index_col=0)
    qm7_labels = [x.split('/')[-1].split('.xyz')[0] for x in qm7_xyz]
    qm7_energy = np.array([float(qm7_props[qm7_props['file'] == label]['energy / Ha']) for label in qm7_labels])

    for i, mol_ncharges in enumerate(qm7_ncharges):
        for ncharge in mol_ncharges:
            qm7_energy[i] -= atom_energy_coeffs[ncharge]

    sigma = 100
    l2reg = 1e-7

    train_sizes, maes, std = learning_curve(qm7_reps, qm7_N, qm7_energy,
                                            np.array([target_rep]), 
                                            np.array([target_N]),
                                            np.array([target_energy]))
    np.savez('learning_curve_10_slatm.npz', train_sizes=train_sizes, maes=maes, std=std)
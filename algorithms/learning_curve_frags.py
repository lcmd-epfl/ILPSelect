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

def learning_curve(X, atoms, y, X_test, atoms_test, y_test, CV=5, ordered=False):
    train_fractions = np.logspace(-1, 0, num=5, endpoint=True)
    train_sizes = [int(len(y)*x) for x in train_fractions]

    if ordered == False:
        maes = np.zeros((CV, 5))
        ints = np.arange(len(X))
        for i in range(CV):
            print("Shuffle training data iter...",i+1,"/",CV)
            np.random.seed(i)
            np.random.shuffle(ints)
            X = X[ints]
            atoms = atoms[ints]
            y = y[ints]

            # opt hypers
            sigmas = [1,10,100,1e3]
            l2regs = [1e-10, 1e-7, 1e-4]
            maes_hypers = np.zeros((len(sigmas), len(l2regs)))

            for i, sigma in enumerate(sigmas):
                for j, l2reg in enumerate(l2regs):
                    mae = KRR(X_sub, atoms_sub, y_sub, X_test, atoms_test, y_test, 
                            sigma=sigma, l2reg=l2reg)
                    maes_hypers[i,j] = mae
            min_i, min_j = np.unravel_index(np.argmin(maes_hypers, axis=None), maes_hypers.shape)
            min_sigma = sigmas[min_i]
            min_l2reg = l2regs[min_j]

            print("Opt hypers", min_sigma, min_l2reg)

            for j, train_size in enumerate(train_sizes):
                print("train size", train_size)
                X_train = X[:train_size]
                y_train = y[:train_size]
                atoms_train = atoms[:train_size]

                mae = KRR(X_train, atoms_train, y_train,
                                            X_test, atoms_test, y_test, sigma=min_sigma,
                                            l2reg=min_l2reg)
                maes[i,j] = mae

        mean_maes = np.mean(maes, axis=0)
        stdev = np.std(maes, axis=0)

    elif ordered == True:
        # opt hypers
        sigmas = [1,10,100,1e3]
        l2regs = [1e-10, 1e-7, 1e-4]
        maes_hypers = np.zeros((len(sigmas), len(l2regs)))

        for i, sigma in enumerate(sigmas):
            for j, l2reg in enumerate(l2regs):
                mae = KRR(X_train, atoms_train, y_train, X_test, atoms_test, y_test, 
                        sigma=sigma, l2reg=l2reg)
                maes_hypers[i,j] = mae
        min_i, min_j = np.unravel_index(np.argmin(maes_hypers, axis=None), maes_hypers.shape)
        min_sigma = sigmas[min_i]
        min_l2reg = l2regs[min_j]

        print("Opt hypers", min_sigma, min_l2reg)

        mean_maes = np.zeros(5)
        for j, train_size in enumerate(train_sizes):
            print("train size", train_size)
            X_train = X[:train_size]
            y_train = y[:train_size]
            atoms_train = atoms[:train_size]

            mae = KRR(X_train, atoms_train, y_train,
                                        X_test, atoms_test, y_test, sigma=min_sigma,
                                        l2reg=min_l2reg)
            mean_maes[j] = mae
        stdev = np.zeros(5)

    return train_sizes, mean_maes, stdev


if __name__ == "__main__":
    with open("atom_energy_coeffs.pickle", "rb") as f:
        atom_energy_coeffs = pickle.load(f)

    props = pd.read_csv("opt-amons-penicillin-target/energies.csv", names=['file', 'energy'])
    target_energy = float(props[props['file'] == 'penicillin.xyz']['energy'])
    target_mol = qml.Compound("opt-amons-penicillin-target/penicillin.xyz")
    for ncharge in target_mol.nuclear_charges:
        target_energy -= atom_energy_coeffs[ncharge]

    frag_indices = np.load('local_environment.npy', allow_pickle=True)
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

    train_sizes, maes, std = learning_curve(qm7_reps, qm7_nat, qm7_energy,
                                            np.array([target_rep]), 
                                            np.array([target_nat]),
                                            np.array([target_energy]),
                                            ordered=True)
    np.savez('learning_curve_frags_cps_local.npz', train_sizes=train_sizes, maes=maes, std=std)

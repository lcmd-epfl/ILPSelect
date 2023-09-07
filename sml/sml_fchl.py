import qml
import numpy as np
from qml.math import cho_solve
import pandas as pd
import random
import pickle
import pdb
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from qml.kernels import get_global_kernel
import os

random.seed(42)
np.random.seed(42)
#matplotlib font size 
plt.rcParams.update({'font.size': 18})

Ha2kcal =  627.5 
def krr(kernel, properties, l2reg=1e-9):
    alpha = cho_solve(kernel, properties, l2reg=l2reg)
    return alpha

def get_kernel(X1, X2, charges1, charges2, sigma=1):
    K = qml.kernels.get_local_kernel(X1, X2, charges1, charges2, sigma)
    return K

def train_model(X_train, atoms_train, y_train, sigma=1, l2reg=1e-9):
    K_train = get_kernel(X_train, X_train, atoms_train, atoms_train, sigma=sigma)
    alpha_train = krr(K_train, y_train, l2reg=l2reg)
    return alpha_train

def train_predict_model(X_train, atoms_train, y_train, X_test, atoms_test, y_test, sigma=1, l2reg=1e-9):
    alpha_train = train_model(X_train, atoms_train, y_train, sigma=sigma, l2reg=l2reg)

    K_test = get_kernel(X_train, X_test, atoms_train, atoms_test)
    y_pred = np.dot(K_test, alpha_train)
    mae = np.abs(y_pred - y_test)[0]
    return mae, y_pred


def opt_hypers(X_train, atoms_train, y_train):
    sigmas = [0.25,0.5,0.75, 1e0, 1e1, 1.25, 1.5]
    l2regs = [1e-7, 1e-6, 1e-4]
    
    n_folds = 5
    kf = KFold(n_splits=n_folds)
    
    maes = np.zeros((len(sigmas), len(l2regs)))
    
    for i, sigma in enumerate(sigmas):
        for j, l2reg in enumerate(l2regs):
            fold_maes = []
            for train_index, val_index in kf.split(X_train):
                X_train_fold, X_val_fold = X_train[train_index], X_train[val_index]
                atoms_train_fold, atoms_val_fold = atoms_train[train_index], atoms_train[val_index]
                y_train_fold, y_val_fold = y_train[train_index], y_train[val_index]
                
                mae, _ = train_predict_model(X_train_fold, atoms_train_fold, y_train_fold,
                                             X_val_fold, atoms_val_fold, y_val_fold,
                                             sigma=sigma, l2reg=l2reg)
                fold_maes.append(mae)
            
            avg_mae = np.mean(fold_maes)
            print('sigma', sigma, 'l2reg', l2reg, 'avg mae', avg_mae)
            maes[i, j] = avg_mae
            
    min_j, min_k = np.unravel_index(np.argmin(maes, axis=None), maes.shape)
    min_sigma = sigmas[min_j]
    min_l2reg = l2regs[min_k]
    print('min avg mae', maes[min_j, min_k], 'for sigma=', min_sigma, 'and l2reg=', min_l2reg)
    
    return min_sigma, min_l2reg

def get_representations(mols, max_natoms=None, elements=None):
    if max_natoms is None:
        max_natoms = max([len(mol.nuclear_charges) for mol in mols])
    if elements is None:
        elements = np.unique(np.concatenate([(mol.nuclear_charges) for mol in mols]))

    reps = np.array([qml.representations.generate_fchl_acsf(mol.nuclear_charges, 
                                                mol.coordinates,
                                                elements=elements,
                                                gradients=False,
                                                pad=max_natoms)
            for mol in mols])
    nuclear_charges = np.array([mol.nuclear_charges for mol in mols])
    return reps, nuclear_charges

def get_ranking_rep(X_train, X_target):
    distances = np.linalg.norm(np.sum(X_train, axis=2) - np.sum(X_target, axis=2) , axis=1)
    sorted_indices = np.argsort(distances)
    return sorted_indices


if __name__ == '__main__':

    NEW_FIT, PLOT = True, True
    if os.uname()[1] == "voy":
        ALL_TARGETS = pd.read_csv("/home/jan/projects/molekuehl/cluster/targets/targets.csv")
    else:
        ALL_TARGETS = pd.read_csv("/home/weinreic/exe/molekuehl/cluster/targets/targets.csv")
    TARGETS_XYZ, TARGETS_y = ALL_TARGETS["xyz"].values, ALL_TARGETS["energies"].values



    if NEW_FIT:
        #here everything in hartree units
        with open('atom_energy_coeffs.pickle', 'rb') as f:
            atom_energy_coeffs = pickle.load(f)



        for xyz_target, y_target in zip(TARGETS_XYZ, TARGETS_y):
            print("Target:", xyz_target)
            target_name = xyz_target.split(".")[0]
            #get the name of the current host
            if os.uname()[1] == "voy":
                TARGET_PATH = f"/home/jan/projects/molekuehl/sml/targets/{xyz_target}"
                FRAGMENTS_PATH = "/home/jan/projects/molekuehl/qm7"
                FRAG_y = pd.read_csv(f"{FRAGMENTS_PATH}/energies.csv")
            else:
                TARGET_PATH = f"/home/weinreic/exe/molekuehl/cluster/targets/{xyz_target}"
                FRAGMENTS_PATH = "/home/weinreic/sml/qm7"
                FRAG_y = pd.read_csv(f"{FRAGMENTS_PATH}/energies_qm7.csv")
            #FRAGMENTS_PATH = "./qm7"
            #randomly shuffle the data
            FRAG_y = FRAG_y.sample(frac=1, random_state=42)
            xyzs, y_train =FRAG_y["file"].values, FRAG_y["energy / Ha"].values



            mols       = np.array([qml.Compound(f"{FRAGMENTS_PATH}/{x}.xyz") for x in xyzs])
            target_mol = qml.Compound(TARGET_PATH)
            target_nat = len(target_mol.nuclear_charges)
            for ncharge in target_mol.nuclear_charges:
                y_target -= atom_energy_coeffs[ncharge]

            X, Q = get_representations(mols, max_natoms=len(target_mol.coordinates))
            qm7_nat = np.array([len(x) for x in Q])
            elements_qm7 = np.unique(np.concatenate([(mol.nuclear_charges) for mol in mols]))
            

            for i, mol_ncharges in enumerate(Q):
                for ncharge in mol_ncharges:
                    y_train[i] -= atom_energy_coeffs[ncharge]


            X_target, Q_target = get_representations([target_mol], max_natoms=len(target_mol.coordinates), elements=elements_qm7)
            
            
            N = [2**i for i in range(4, 13)][:11]
            
            #N.append(len(X))
            mae_sml = []


            Q = np.array(Q)
            opt_ranking = get_ranking_rep(X,X_target)

            
            for n in N:
                ranking = opt_ranking[:n]
                min_sigma, min_l2reg =  opt_hypers(X[ranking], Q[ranking], y_train[ranking])
                print(min_sigma, min_l2reg)
                mae, y_pred          = train_predict_model(X[ranking], Q[ranking], y_train[ranking], X_target, Q_target, y_target, sigma=min_sigma, l2reg=min_l2reg)
                mae_sml.append(mae)
                print("SML", n, mae)

            mae_sml = np.array(mae_sml)

            #five fold cross validation
            CV = 5
            all_maes_random = []
            kf = KFold(n_splits=CV, shuffle=True, random_state=130)

            for i, (train_index, test_index) in enumerate(kf.split(X)):
                maes_random = []

                for n in N:

                    min_sigma, min_l2reg =opt_hypers(X[train_index][:n], Q[train_index][:n], y_train[train_index][:n])
                    print(min_sigma, min_l2reg)
                    mae, y_pred          = train_predict_model(X[train_index][:n], Q[train_index][:n], y_train[train_index][:n], X_target, Q_target, y_target, sigma=min_sigma, l2reg=min_l2reg)
                    
                    maes_random.append(mae)
                    print("Random", n, mae)

                all_maes_random.append(maes_random)

            all_maes_random = np.array(all_maes_random)
            np.savez(f'./results/learning_curve_sml_{target_name}.npz', train_sizes=N, all_maes_random=all_maes_random, mae_sml=mae_sml, ranking_xyz=xyzs[opt_ranking])
    
    if PLOT:
        for xyz_target in TARGETS_XYZ:
            target_name = xyz_target.split(".")[0]
            LEARNING_CURVE = np.load(f'./results/learning_curve_sml_{target_name}.npz')
            MEAN_RANDOM, STD_RANDOM = np.mean(LEARNING_CURVE['all_maes_random'], axis=0)*Ha2kcal, np.std(LEARNING_CURVE['all_maes_random'], axis=0)*Ha2kcal
            SML = LEARNING_CURVE['mae_sml']*Ha2kcal
            N = LEARNING_CURVE['train_sizes']
            #create figure and axis
            fig, ax = plt.subplots(figsize=(11,6))
            #plot learning curve random with std as error bars
            ax.errorbar(N, MEAN_RANDOM, yerr=STD_RANDOM, fmt='o-', label='Random')
            #plot learning curve SML
            ax.plot(N, SML, 'o-', label='SML')
            #set axis labels
            ax.set_xlabel('Training set size')
            ax.set_ylabel('MAE [kcal/mol]')
            #set log scale on x axis
            ax.set_xscale('log')
            #set log scale on y axis
            ax.set_yscale('log')
            #legend
            ax.legend()
            #save figure
            #turn minor ticks off 
            ax.minorticks_off()
            #make x ticks as N
            ax.set_xticks(N)
            ax.set_xticklabels(N)
            #grid on
            ax.grid()
            #save figure
            fig.savefig(f'./results/learning_curve_sml_{target_name}.png', dpi=300)
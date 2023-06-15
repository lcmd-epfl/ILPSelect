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

def opt_hypers(X_train, atoms_train, y_train, X_test, atoms_test, y_test):
    sigmas = [0.5, 0.75, 1, 1.25]
    l2regs = [1e-10, 1e-7, 1e-4]
    
    maes = np.zeros((len(sigmas), len(l2regs)))
    for i, sigma in enumerate(sigmas):
        for j, l2reg in enumerate(l2regs):
            mae, y_pred = train_predict_model(X_train, atoms_train, y_train, X_test, atoms_test,
                                             y_test, sigma=sigma, l2reg=l2reg)
            print('sigma', sigma, 'l2reg', l2reg, 'mae', mae)
            maes[i,j] = mae
            
    min_j, min_k = np.unravel_index(np.argmin(maes, axis=None), maes.shape)
    min_sigma = sigmas[min_j]
    min_l2reg = l2regs[min_k]
    print('min mae', maes[min_j, min_k], 'for sigma=', min_sigma, 'and l2reg=', min_l2reg)
    return min_sigma, min_l2reg

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


def get_ranking(X, X_target, Q, Q_target):
    K = get_kernel(X, X_target, Q, Q_target, sigma=1)
    return np.argsort(K)[::-1]


if __name__ == '__main__':

    NEW_FIT, PLOT = False, True
    ALL_TARGETS = pd.read_csv("./targets/targets.csv")
    TARGETS_XYZ, TARGETS_y = ALL_TARGETS["xyz"].values, ALL_TARGETS["energies"].values



    if NEW_FIT:
        #here everything in hartree units
        with open('atom_energy_coeffs.pickle', 'rb') as f:
            atom_energy_coeffs = pickle.load(f)



        for xyz_target, y_target in zip(TARGETS_XYZ, TARGETS_y):
            print("Target:", xyz_target)
            target_name = xyz_target.split(".")[0]
            TARGET_PATH = f"./targets/{xyz_target}"
            FRAGMENTS_PATH = "/home/jan/projects/molekuehl/qm7"
            FRAG_y = pd.read_csv(f"{FRAGMENTS_PATH}/energies_qm7.csv")
            #randomly shuffle the data
            FRAG_y = FRAG_y.sample(frac=1, random_state=42)
            xyzs, y_train =FRAG_y["file"].values, FRAG_y["energy / Ha"].values



            mols       = np.array([qml.Compound(f"{FRAGMENTS_PATH}/{x}.xyz") for x in xyzs])
            target_mol = qml.Compound(TARGET_PATH)
            target_nat = len(target_mol.nuclear_charges)
            for ncharge in target_mol.nuclear_charges:
                y_target -= atom_energy_coeffs[ncharge]

            X, Q = get_representations(mols, params=None)
            qm7_nat = np.array([len(x) for x in Q])

            for i, mol_ncharges in enumerate(Q):
                for ncharge in mol_ncharges:
                    y_train[i] -= atom_energy_coeffs[ncharge]


            X_target, Q_target = get_representations([target_mol], params=None)
            
            
            N = [2**i for i in range(4, 13)][:7]
            #N.append(len(X))
            mae_sml = []
            #SML
            opt_ranking = get_ranking(X, X_target, Q, Q_target)[0]
            
            for n in N:
                ranking = opt_ranking[:n]
                min_sigma, min_l2reg = opt_hypers(X[ranking], Q[ranking], y_train[ranking], X_target, Q_target, y_target)
                mae, y_pred          = train_predict_model(X[ranking], Q[ranking], y_train[ranking], X_target, Q_target, y_target, sigma=min_sigma, l2reg=min_l2reg)
                mae_sml.append(mae)
                print("SML", n, mae)

            mae_sml = np.array(mae_sml)

            #five fold cross validation
            CV = 5
            all_maes_random = []
            for i in range(CV):
                maes_random = []
                ints = np.arange(len(X))
                print("Shuffle training data iter...",i+1,"/",CV)
                np.random.seed(i)
                np.random.shuffle(ints)
                X_sub = X[ints]
                Q_sub = Q[ints]
                y_sub = y_train[ints]

                for n in N:

                    min_sigma, min_l2reg = opt_hypers(X_sub[:n], Q_sub[:n], y_sub[:n], X_target, Q_target, y_target)
                    mae, y_pred          = train_predict_model(X_sub[:n], Q_sub[:n], y_sub[:n], X_target, Q_target, y_target, sigma=min_sigma, l2reg=min_l2reg)
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

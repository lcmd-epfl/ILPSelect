import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import matplotlib
import pickle
from ase.io import read
matplotlib.rcParams.update({'font.size': 14})
import numpy as np
# y energies offset
with open(f"data/atom_energy_coeffs.pickle", "rb") as f:
    # in Ha
    atom_energy_coeffs = pickle.load(f)

def correct_energies(mol, energy, atom_energy_coeffs):
    atomtypes = mol.get_atomic_numbers()
    for atom in atomtypes:
        energy -= atom_energy_coeffs[atom]
    return energy

qm7_files = pd.read_csv('qm7/energies.csv')['file'].tolist()
mols = [read('qm7/'+x+'.xyz') for x in qm7_files]
print(len(mols))
energies = pd.read_csv('qm7/energies.csv')['energy / Ha'].to_numpy()
print(energies.size)
y_train = [correct_energies(mols[i], energies[i], atom_energy_coeffs) * 627.503 for i in range(len(energies))]
print(len(y_train))

test_df = pd.read_csv('targets/energies.csv')
target_names_qm9= [
    "121259.xyz",
    "12351.xyz",
    "35811.xyz",
    "85759.xyz",
    "96295.xyz",
    "5696.xyz",
    "31476.xyz",
    "55607.xyz",
    "68076.xyz",
    "120425.xyz"]
qm9_files = test_df[test_df['file'].isin(target_names_qm9)]['file'].tolist()
energies_qm9 = test_df[test_df['file'].isin(target_names_qm9)]['energy / Ha'].to_numpy()
mols = [read('targets/'+x) for x in qm9_files]
y_test_qm9 = [correct_energies(mols[i], energies_qm9[i], atom_energy_coeffs) * 627.503 for i in range(len(energies_qm9))]

target_names_drugs= [
    "sildenafil.xyz",
    "penicillin.xyz",
    "troglitazone.xyz",
    "imatinib.xyz",
    "pemetrexed.xyz",
    "oxycodone.xyz",
    "pregabalin.xyz",
    "apixaban.xyz",
    "salbutamol.xyz",
    "oseltamivir.xyz",
]
d_files = test_df[test_df['file'].isin(target_names_drugs)]['file'].tolist()
energies_d = test_df[test_df['file'].isin(target_names_drugs)]['energy / Ha'].to_numpy()
mols = [read('targets/'+x) for x in d_files]
y_test_drugs = [correct_energies(mols[i], energies_d[i], atom_energy_coeffs) * 627.503 for i in range(len(energies_d))]

fig, ax = plt.subplots(nrows=1, ncols=1)
#sns.kdeplot(np.array(energies), fill=True, label='QM7', alpha=0.7)
#sns.kdeplot(np.array(energies_qm9), fill=True, label='QM9(9)', alpha=0.7)
#sns.kdeplot(np.array(energies_d), fill=True, label='Drugs', alpha=0.7)
sns.histplot(np.array(y_train), fill=True, label='QM7', alpha=0.20, stat='density', binwidth=20)
sns.histplot(np.array(y_test_qm9), fill=True, label='QM9(9)', alpha=0.20, stat='density', binwidth=20)
sns.histplot(np.array(y_test_drugs), fill=True, label='Drugs', alpha=0.20, stat='density', binwidth=20)
plt.legend(loc='upper left')
sns.kdeplot(np.array(y_train), fill=False, label='QM7', alpha=1.0, cut=0.1, bw_adjust=1, legend=False)
sns.kdeplot(np.array(y_test_qm9), fill=False, label='QM9(9)', alpha=1.0, cut=1, bw_adjust=1, legend=False)
sns.kdeplot(np.array(y_test_drugs), fill=False, label='Drugs', alpha=1.0, cut=0.5, bw_adjust=.5, legend=False)
ax.set_ylabel("Density")
#ax.set_xlabel("Total energy (PBE0-D3/def2-SVP) / Ha")
ax.set_xlabel(r"$\hat{E}$ (PBE0-D3/def2-SVP) [kcal/mol]")
plt.tight_layout()
plt.savefig('OOD.pdf')
plt.show()

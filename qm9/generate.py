# Script to generate the xyz files and the energies.csv from the qm9_data.npz given by Jan.
# qm9_data.npz is too large to push to the repository.
# %%
import numpy as np
import pandas as pd

data = np.load("../cluster/data/qm9_data.npz", allow_pickle=True)
data_indices = data["index"]  # data["index"] has holes
data_elements = data["elements"]
data_u0 = data["U0"]
# data_h_atomization = data["H_atomization"]

# taken from https://springernature.figshare.com/articles/dataset/Atomref_Reference_thermochemical_energies_of_H_C_N_O_F_atoms_/1057643?backTo=/collections/Quantum_chemistry_structures_and_properties_of_134_kilo_molecules/978904
U0_atom_coeffs = {
    "H": -0.500273,
    "C": -37.846772,
    "N": -54.583861,
    "O": -75.064579,
    "F": -99.718730,
}

# %%

df = pd.DataFrame(
    {
        "file": data_indices,
        "energy / Ha": data_u0,
        "elements": data_elements
        # "atomization energy / Ha": data_h_atomization,
    }
)

df["sum_atom_U0"] = df["elements"].apply(
    lambda x: sum(U0_atom_coeffs[charge] for charge in x)
)

df["atomization energy / Ha"] = df["energy / Ha"] - df["sum_atom_U0"]

df.drop(columns={"elements", "sum_atom_U0"})
# %%

df.to_csv("energies.csv")

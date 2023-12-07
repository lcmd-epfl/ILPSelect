# Script to generate the xyz files and the energies.csv from the qm9_data.npz given by Jan.
# qm9_data.npz is too large to push to the repository.
#%%
import numpy as np
import pandas as pd

data=np.load("../cluster/data/qm9_data.npz", allow_pickle=True)
data_indices=data["index"] # data["index"] has holes
data_coordinates=data["coordinates"]
data_elements = data["elements"]
data_charges = data["charges"]
data_u0 = data["U0"]
data_h_atomization = data["H_atomization"]

# %%

df = pd.DataFrame({"file": data_indices, "energy / Ha":data_u0, "atomization energy / Ha": data_h_atomization})
#df["file"] = df["file"].apply(lambda x: "qm9_"+str(x))

df.to_csv("energies.csv")
# %%

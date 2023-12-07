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

# %%

for i in range(len(data_indices)):
    charges = data_charges[i]
    coordinates = data_coordinates[i]
    elements = data_elements[i]
    with open(f"qm9_{i}.xyz", 'w') as xyz_file:
        xyz_file.write("%d\n%s\n" % (len(charges), ""))
        for j in range(len(charges)):
            xyz_file.write("{:4} {:11.6f} {:11.6f} {:11.6f}\n".format(
                elements[j], coordinates[j][0], coordinates[j][1], coordinates[j][2]))
            
#%%

df = pd.DataFrame({"file": data_indices, "energy / Ha":data_u0})
#df["file"] = df["file"].apply(lambda x: "qm9_"+str(x))

df.to_csv("energies.csv")
# %%

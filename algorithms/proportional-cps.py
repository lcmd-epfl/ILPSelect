# Closest point sampling: picks fragments from database
# that such that some atom is closest to some atom of the target.

# The atom of the target that is best matched is recorded in the subset selection,
# and the density of each atom type much be inversely proportional to its density
# in the target.

import fragments
import pandas as pd
import numpy as np
import sys
import time

### change this accordingly
repfolder='../representations/'
outfolder='../out/'
###

data=np.load(repfolder+"qm7_SLATM_local_data-renamed.npz", allow_pickle=True)
target=np.load(repfolder+"pruned-penicillin_SLATM_local_data.npz", allow_pickle=True)

size_target=len(target['ncharges'])
size_data=len(data['labels'])

# fills out following frame
# only needs to be done once (to_csv, read_csv)!
frame=pd.DataFrame({
    "target_index":[],
    "ncharge":[],
    "fragment_index":[],
    "atom_index":[],
    "distance":[]
    })

for F in range(size_data):
    print(f"{F} / {size_data}")
    size_fragment=len(data['ncharges'][F])
    S=[(i,j) for i in range(size_target) for j in range(size_fragment) if target['ncharges'][i] == data['ncharges'][F][j]]
    
    k=0
    for (i,j) in S:
        k+=1
        print(k, len(S))
        ncharge=target['ncharges'][i]
        dist=np.linalg.norm(target['rep'][i] - data['reps'][F][j])
        newline=pd.DataFrame({
            "target_index":[i],
            "ncharge":[ncharge],
            "fragment_index":[F],
            "atom_index":[j],
            "distance":[dist]
            })
        frame=pd.concat([frame,newline])

print(frame)
frame.to_csv(outfolder+"qm7_SLATM_local_frame.csv")

# for each atom-type, compute the inverse proportion p
# and sample the first p*1000 in ascending distance from the dataframe
ncharges = set(target['ncharges'])
proportions={i:0 for i in ncharges}
for i in range(size_target):
    proportions[target['ncharges'][i]]+=1

for i in ncharges:
    proportions[i]**=-1

norm_constant=sum([proportions[i] for i in ncharges])

for i in ncharges:
    proportions[i]/=norm_constant

print(proportions)

frame=frame.set_index("ncharge")

selection=pd.DataFrame()
for i in ncharges:
    p=proportions[i]
    best=frame.loc[i].sort_values("distance").head(np.ceil(p*1000))
    selection=pd.concat([selection,best])

print(selection)
selection.to_csv(outfolder+"qm7_SLATM_local_proportional_cps.csv")

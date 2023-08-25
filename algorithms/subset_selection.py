import fragments
import pandas as pd
import numpy as np
import sys
import time

### change this accordingly
repfolder='../sml/results/'
outfolder='../sml/results/'
###

M=fragments.model(repfolder+"data_qm7.npz", repfolder+"sildenafil.npz", scope="local_vector", verbose=True)

# sets up model and writes to a file. Only needed once.
M.setup(penalty_constant=0, duplicates=1)
M.savemodel(outfolder+'comp.mps')

# reads files and changes the penalty constant
#pen=float(sys.argv[1])
#M.readmodel("../models/SLATM-local-5.mps")
#M.changepenalty(pen)

# reads already found combinations to remove then (if we want to continue previous optimization for example)
# df=pd.read_csv(outfolder+"newsolutions"+str(pen)+".csv")
# M.add_forbidden_combinations(df['Fragments'].apply(eval))

# ordered_frags is ordered in groups of 100 as well since objective values cannot be compared
ordered_frags=[]
# optimize with callback
M.optimize(number_of_solutions=100, PoolSearchMode=2, timelimit=4*3600, poolgapabs=35, callback=True, objbound=30, number_of_fragments=1024)

print(M.visitedfragments)

df=pd.DataFrame(M.solutions)
df.to_csv(outfolder+"comp-sol.csv")

# sorts solutions found by objective value
# and saves fragments to file
sorteddf=df.sort_values('Value')['Fragments']
for e in sorteddf:
    for f in e:
        if not f in ordered_frags:
            ordered_frags.append(f)

np.save(outfolder+'comp-rank.npy', ordered_frags)


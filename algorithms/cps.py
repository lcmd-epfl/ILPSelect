import fragments
import pandas as pd
import numpy as np
import sys
import time

### change this accordingly
repfolder='../representations/'
outfolder='../out/'
###

M=fragments.model(repfolder+"qm7_SLATM_local_data-renamed.npz", repfolder+"pruned-penicillin_SLATM_local_data.npz", scope="local_vector", verbose=True)

# sets up model and writes to a file. Only needed once.
#M.setup(penalty_constant=10, duplicates=1)
#M.savemodel(outfolder+'out.mps')

# reads files and changes the penalty constant
M.readmodel("../models/SLATM-local-5.mps")
M.changepenalty(0)

# add cps constraint
M.add_cps_constraint()

# optimize
CPS=[]
CPS_labels=[]
for i in range(3):
    M.optimize(number_of_solutions=1, PoolSearchMode=1, timelimit=4*3600)
    d=M.output()
    print(d)
    M.remove_fragments(d['FragmentsID'])
    CPS.append(d['FragmentsID'][0][0])
    CPS_labels.append(d['Fragments'][0][0])

print(CPS)
print(CPS_labels)
np.save(f"{outfolder}CPS_{time.strftime('%Y-%m-%d')}.npy", CPS)
np.save(f"{outfolder}CPS_labels_{time.strftime('%Y-%m-%d')}.npy", CPS_labels)

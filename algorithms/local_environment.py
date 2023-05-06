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

M.readmodel("../models/SLATM-local-5.mps")
M.changepenalty(0)

ordered_frags=[]
while len(ordered_frags) < 1100:
    # optimize with callback
    M.optimize(number_of_solutions=1, PoolSearchMode=0, timelimit=300)

    d = M.output()
    fragment_array=d["FragmentsID"]
    M.remove_fragments(fragment_array)
    
    fragment_ids=fragment_array[0]
    ordered_frags+=fragment_ids

np.save(outfolder+"local_environment.npy", ordered_frags)

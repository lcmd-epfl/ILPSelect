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
pen=float(sys.argv[1])
M.readmodel("../models/SLATM-local-5.mps")
#M.changepenalty(pen)

# reads already found combinations to remove then (if we want to continue previous optimization for example)
# df=pd.read_csv(outfolder+"newsolutions"+str(pen)+".csv")
# M.add_forbidden_combinations(df['Fragments'].apply(eval))

# reduces penalty every 100 fragments (from 5 to pen)
# ordered_frags is ordered in groups of 100 as well since objective values cannot be compared
ordered_frags=[]
for i in range(10):
    newpen=5-i/9*(5-pen)
    M.changepenalty(newpen)
    
    # optimize with callback
    M.optimize(number_of_solutions=100, PoolSearchMode=1, timelimit=4*3600, poolgapabs=35, callback=True, objbound=30, number_of_fragments=100*(i+1))

    print(M.visitedfragments)

    df=pd.DataFrame(M.solutions)
    df.to_csv(outfolder+"solutions"+str(pen)+"-"+time.strftime("%Y%m%d-%H%M%S")+".csv")

    # sorts solutions found by objective value
    # and saves fragments to file
    sorteddf=df.sort_values('Value')['Fragments']
    for e in sorteddf:
        for f in e:
            if not f in ordered_frags:
                ordered_frags.append(f)

    np.save(outfolder+'frag'+str(pen)+"-"+time.strftime("%Y%m%d-%H%M%S")+'.npy', ordered_frags)


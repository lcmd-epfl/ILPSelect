import fragments
import pandas as pd
import numpy as np
import sys

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
M.readmodel(outfolder+'out.mps')
M.changepenalty(pen)

M.optimize(number_of_solutions=10, PoolSearchMode=1, timelimit=300, poolgapabs=35, callback=True, objbound=35, number_of_fragments=10)

print(M.visitedfragments)
np.save(outfolder+"newfrag"+str(pen)+".npy", np.array(M.visitedfragments))

"""
solutions={"Fragments":[], "Value":[]}
for i in range(24):
    I=M.randomsubset(0.821)
    print("Iteration", i)
    print("Dataset of size", len(I))
    M.optimize(number_of_solutions=10, PoolSearchMode=1, timelimit=600, poolgapabs=30)
    M.output()
    d=M.SolDict
    M.add_forbidden_combinations(d['FragmentsID'])
    for k in range(len(d['FragmentsID'])):
        print(d["FragmentsID"][k])
        solutions["Fragments"].append(d["FragmentsID"][k])
        solutions["Value"].append(d["ObjValWithPen"][k])

df=pd.DataFrame(solutions)
df.to_csv(outfolder+"solutions" + str(pen) + ".csv")

#sorts solutions found by objective value
sorteddf=df.sort_values('Value')['Fragments']
#adds all fragments in order to fullarray, allowing duplicates
fullarray=[]
for e in sorteddf:
    for f in e:
        fullarray.append(f)

#creates dict from keys of fullarray, keeping the first occurence of each fragment only, and hence keeping order of appearance
ordered_frags=list(dict.fromkeys(fullarray))
np.save(outfolder+'ordered_fragments'+str(pen)+'.npy', ordered_frags)
"""

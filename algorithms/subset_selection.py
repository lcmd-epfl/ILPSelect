import fragments
import pandas as pd
import numpy as np

### change this accordingly
repfolder='/scratch/haeberle/'
outfolder='/scratch/haeberle/out/'
###

M=fragments.model(repfolder+"qm7_FCHL_global_data-renamed.npz", repfolder+"penicillin_FCHL_global_data.npz", scope="global_vector", verbose=True)
M.setup(penalty_constant=0, duplicates=1, nthreads=15, poolgapabs=50)

solutions={"Fragments":[], "Value":[]}
for i in range(35):
    I=M.randomsubset(0.821)
    print("Iteration", i)
    print("Dataset of size", len(I))
    M.optimize(number_of_solutions=50, PoolSearchMode=1, timelimit=600)
    M.output()
    d=M.SolDict
    M.add_forbidden_combinations(d['FragmentsID'])
    for k in range(len(d['FragmentsID'])):
        print(d["FragmentsID"][k])
        solutions["Fragments"].append(d["FragmentsID"][k])
        solutions["Value"].append(d["ObjValWithPen"][k])

df=pd.DataFrame(solutions)
df.to_csv(outfolder+"solutions.csv")

#sorts solutions found by objective value
sorteddf=df.sort_values('Value')['Fragments']
#adds all fragments in order to fullarray, allowing duplicates
fullarray=[]
for e in sorteddf:
    for f in e:
        fullarray.append(f)

#creates dict from keys of fullarray, keeping the first occurence of each fragment only, and hence keeping order of appearance
ordered_frags=list(dict.fromkeys(fullarray))
np.save(outfolder+'ordered_fragments.npy', ordered_frags)

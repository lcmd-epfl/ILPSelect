import fragments
import pandas as pd
import numpy as np

M=fragments.model("/scratch/haeberle/qm7_FCHL_global_data-renamed.npz", "/scratch/haeberle/penicillin_FCHL_global_data.npz", scope="global_vector", verbose=True)
M.setup(penalty_constant=0, duplicates=1)

solutions={"Fragments":[], "Value":[]}
for i in range(30):
    I=M.randomsubset(0.821)
    print("Iteration", i)
    print("Dataset of size", len(I))
    M.optimize(number_of_solutions=50, PoolSearchMode=2, timelimit=600)
    M.output()
    d=M.SolDict
    M.add_forbidden_combinations(d['FragmentsID'])
    for k in range(len(d['FragmentsID'])):
        print(d["FragmentsID"][k])
        solutions["Fragments"].append(d["FragmentsID"][k])
        solutions["Value"].append(d["ObjValWithPen"][k])

df=pd.DataFrame(solutions)
df.to_csv("/scratch/haeberle/out/solutions.csv")

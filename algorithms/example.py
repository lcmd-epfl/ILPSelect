import fragments
import pandas as pd
import numpy as np

M=fragments.model("../representations/qm7_FCHL_global_data-renamed.npz", "../representations/penicillin_FCHL_global_data.npz", scope="global_vector", verbose=True)
M.setup(penalty_constant=0, duplicates=1)

F=set()
i=0
while i<5 and len(F)<600:
    i+=1
    I=M.randomsubset(0.821)
    print("Iteration", i)
    print("Dataset of size", len(I))
    M.optimize(number_of_solutions=10, PoolSearchMode=1, timelimit=10*60)
    M.output()
    d=M.SolDict
    M.add_forbidden_combinations(d['FragmentsID'])
    for s in d['FragmentsID']:
        print(s)
        for f in s:
            F.add(f)
    print(F, len(F))

np.save("../out/fragments.npy", F)


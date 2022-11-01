import fragments
import pandas as pd
import numpy as np

totaltime=25200 # 7 hours
timeperiter=5*60 # 5 minutes
solperiter=10

timeaftersetup = totaltime - 25*60 # keep 25 mins tops for setup
N=int(timeaftersetup/timeperiter) # number of iterations
print(N)
M=fragments.model("../representations/qm7_FCHL_global_data-renamed.npz", "../representations/penicillin_FCHL_global_data.npz", scope="global_vector", verbose=True)
M.setup(penalty_constant=1e3, duplicates=1)

F=set()
for i in range(N):
    print("Iteration", i+1, "out of", N)
    M.optimize(number_of_solutions=solperiter, PoolSearchMode=2, timelimit=timeperiter)
    M.output()
    d=M.SolDict
    for s in d['FragmentsID']:
        for f in s:
            F.add(f)
    print(F, len(F))
    for i in range(M.SolCount):
        fragsid=d['FragmentsID'][i]
        print(fragsid)
        M.add_forbidden_combination(fragsid)

np.save("../out/fragments.npy", F)

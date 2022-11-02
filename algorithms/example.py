import fragments
import pandas as pd
import numpy as np

totaltime=8*3600 # 8 hours
timeperiter=10*60 # 10 minutes
solperiter=10

timeaftersetup = totaltime - 25*60 # keep 25 mins tops for setup and in-between delay
N=int(timeaftersetup/timeperiter) # number of iterations
print(N)
M=fragments.model("../representations/qm7_FCHL_global_data-renamed.npz", "../representations/penicillin_FCHL_global_data.npz", scope="global_vector", verbose=True)
M.setup(penalty_constant=0, duplicates=1)

F=set()
i=0
while i<N and len(F)<600:
    i+=1
    print("Iteration", i+1, "out of", N)
    M.optimize(number_of_solutions=solperiter, PoolSearchMode=1, timelimit=timeperiter)
    M.output()
    d=M.SolDict
    for s in d['FragmentsID']:
        print(s)
        for f in s:
            F.add(f)
            M.add_forbidden_combination([f])
    print(F, len(F))

np.save("../out/fragments.npy", F)

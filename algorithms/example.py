import fragments
import pandas as pd

M=fragments.model("../representations/qm7_FCHL_global_data-renamed.npz", "../representations/penicillin_FCHL_global_data.npz", "global_vector")
M.setup(penalty_constant=1e3, duplicates=2)
M.optimize(number_of_solutions=20, PoolSearchMode=1)
M.output("../out/penicillin_FCHL_solutions.csv")

d=M.d
F=set()
for s in d['Fragments']:
    for f in s:
        F.add(f)

df=pd.DataFrame(d)
df.to_csv("../out/penicillin_FCHL_fragments.csv")
print(F)
print(len(F))

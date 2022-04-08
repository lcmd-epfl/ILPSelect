import numpy as np
import pandas as pd
import ast

# only for global for now

representation=0 
target=0 
repname=["CM", "SLATM", "SOAP"][representation]
targetname=["qm9", "vitc", "vitd"][target]

d=pd.read_csv("output_"+repname+"_global.csv")
data=np.load("../representations/amons_"+repname+"_global_data.npz", allow_pickle=True)
targetdata=np.load("../representations/target_"+repname+"_global_data.npz", allow_pickle=True)
n=len(d)

print("Reading solutions of representation ", repname)
print("Size of solution pool:", n)

targetcharges=targetdata["target_ncharges"][target]
types=np.unique(targetcharges, return_counts=True)
numbertypes=len(types[0])
print(types)
size_target=len(targetcharges)

sum_sizes_solutions=0
sum_typeexcess=0
for i in range(n):
    frags=ast.literal_eval(d.loc[i]["Fragments"])
    frag_indices=[]
    totalcharges=[]
    for label in frags:
        index=np.where(data[targetname+"_amons_labels"]==label)[0][0]
        charges=data[targetname+"_amons_ncharges"][index]
        totalcharges.append(charges)
        frag_indices.append(index)
        sum_sizes_solutions+=len(charges)
    totalcharges=np.concatenate(totalcharges)
    print(totalcharges)
    for k in range(numbertypes):
        t=types[0][k]
        sum_typeexcess+=np.abs(types[1][k] - np.sum(totalcharges==t))
    print(sum_typeexcess)

print("Type excess: ", sum_typeexcess)
print("Atom excess: ", sum_sizes_solutions - n*size_target)

# todo: connectivity 

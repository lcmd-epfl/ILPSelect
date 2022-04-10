import numpy as np
import pandas as pd
import ast
import gurobipy as gp

representation=0
prefix="_global" # "_global" or empty string "" for local
target=0 # connectivity data only for target 0  
repname=["CM", "SLATM_2", "SLATM_3.5", "SLATM", "SLATM_8", "FCHL_2", "FCHL_3.5", "FCHL_4.8", "FCHL", "SOAP", "aCM"][representation]
targetname=["qm9", "vitc", "vitd"][target]

d=pd.read_csv("output_"+repname+prefix+".csv")
data=np.load("../representations/amons_"+repname+prefix+"_data.npz", allow_pickle=True)
targetdata=np.load("../representations/target_"+repname+prefix+"_data.npz", allow_pickle=True)
CT=targetdata['target_ncharges'][target]
size_database=len(d)

print("Reading solutions of representation", repname)
print("Size of solution pool:", size_database)

targetcharges=targetdata["target_ncharges"][target]
types=np.unique(targetcharges, return_counts=True)
numbertypes=len(types[0])
#print(types)
n=len(targetcharges)

sum_sizes_solutions=0
sum_typeexcess=0
for i in range(size_database):
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
    #print(totalcharges)
    for k in range(numbertypes):
        t=types[0][k]
        sum_typeexcess+=np.abs(types[1][k] - np.sum(totalcharges==t))
    #print(sum_typeexcess)

print("Type excess: ", sum_typeexcess)
sum_atomexcess=sum_sizes_solutions - size_database*n
print("Atom excess: ", sum_atomexcess)

# connectivity only for qm9 (target 0)

targetadj=np.load("../connectivity/qm9_adj.npy", allow_pickle=True)
dataadjs=np.load("../connectivity/qm9_amons_adjs.npz", allow_pickle=True)

# sets up a model to fill target with fragments with connectivity conditions
# returns 0 for success (connected), 1 for failure (unconnected)
def connectivity(frag_indices):
    Z = gp.Model()
    Z.setParam('OutputFlag',0)
    I=[]
    for M in frag_indices:
        CM=data[targetname+"_"+"amons_ncharges"][M]
        m=len(CM)
        I=I+[(i,j,M) for i in range(m) for j in range(n)]# if CM[i] == CT[j]] # if condition excludes j; i always takes all m values

    x=Z.addVars(I, vtype='B')
    Z.addConstrs(x.sum('*',j,'*', '*') == 1 for j in range(n))
    for M in frag_indices:
        CM=data[targetname+"_"+"amons_ncharges"][M]
        m=len(CM)
        # each i of each group is used at most once
        Z.addConstrs(x.sum(i,'*',M) <= 1 for i in range(m))
        Z.addConstrs(x[i,j,M]==0 for i in range(m) for j in range(n) if CM[i]!=CT[j])

    for j in range(n):
        for jj in np.where(targetadj[j]==0)[0]:
            for M in frag_indices:
                CM=data[targetname+"_"+"amons_ncharges"][M]
                m=len(CM)
                for i in range(m):
                    for ii in np.where(dataadjs['adjs'][M][i]==1)[0]:
                        Z.addConstr(x[i,j,M]+x[ii,jj,M]<=1)
    Z.optimize()
    # Z.status is 2 for solution, 3 for unfeasible (other codes are considered unfeasible just in case)
    return Z.status!=2

noncon_count=0 # number of solutions which do not verifiy connectivity (cannot split target in connected components of fragments)
for i in range(size_database):
    frags=ast.literal_eval(d.loc[i]["Fragments"])
    frag_indices=[]
    for label in frags:
        index=np.where(data[targetname+"_amons_labels"]==label)[0][0]
        frag_indices.append(index)
    if(connectivity(frag_indices)):
        noncon_count+=1
        print("Solution number", i, "with fragments", frags, "is not connected.")

print("Number of unconnected solutions: ", noncon_count)
print("Ranking is Type + Atom excess + 10*(number of unconnected solutions):")
print("Rank", sum_typeexcess+sum_atomexcess + 10*noncon_count)

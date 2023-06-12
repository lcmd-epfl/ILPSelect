import numpy as np
import gurobipy as gp
from gurobipy import GRB
import pandas as pd
import matplotlib.pyplot as plt

# takes pandas dataframe and column indices
# outputs np.array of chosen columns
def df_to_nparray(df, indices):
    indices=np.array(indices)
    return np.array(df.take(indices, axis=1))

def add_variables(M, n):
    x=M.addVars(range(n), vtype='B')
    return x

def add_constraints(M, subsetsize, x):
    M.addConstr(x.sum() == subsetsize)
    return 0

def set_objective1(M, points, x):
    n=len(points)
    expr=gp.QuadExpr()
    for i in range(n):
        for j in range(i+1,n):
            norm=np.linalg.norm(points[i]-points[j])
            expr+=norm*x[i]*x[j]
    M.setObjective(expr, GRB.MAXIMIZE)
    return 0

def set_objective2(M, points, x):
    n=len(points)
    obj=M.addVar(vtype='C')
    #maxnorm=np.max([np.linalg.norm(x-y) for x in points for y in points])
    maxnorm=43
    for i in range(n):
        #print(f"{i} / {n}")
        for j in range(i+1,n):
            norm=np.linalg.norm(points[i]-points[j])
            M.addConstr(obj <= norm + maxnorm*(1-x[i]+1-x[j]))
    M.setObjective(obj, GRB.MAXIMIZE)
    return 0

def read_solution(M,n,x):
    chosen_indices=[]
    for i in range(n):
        if x[i].X==1:
            chosen_indices.append(i)
    return np.array(chosen_indices)

def print_solution(points, chosen_indices, filename):
    plt.axes()
    n=len(points)
    r=0
    for i in range(n):
        if i==chosen_indices[r]:
            r=min(len(chosen_indices)-1,r+1)
            circle=plt.Circle((points[i][1],points[i][2]), 0.25, color='red', fill=True, zorder=1)
        else:
            circle=plt.Circle((points[i][1],points[i][2]), 0.25, color='black', fill=True, zorder=0)
        
        plt.gca().add_patch(circle)
    plt.axis('scaled')
    plt.savefig(filename+'-fig.png')
    plt.show()
    return 0

def fps(points, subsetsize):
    # n = number of points
    # d = dimension of space
    n, d=points.shape
    assert subsetsize <= n, "Number of points is smaller than size of subset."
    
    M=gp.Model()
    
    x=add_variables(M, n)

    add_constraints(M,subsetsize,x)

    set_objective2(M, points, x)

    M.optimize()

    assert M.status!=3, "Model is infeasible."
    
    chosen_indices=read_solution(M,n,x)

    return chosen_indices

### change this accordingly
repfolder='../representations/'
outfolder='../out/'
###

data=np.load(repfolder+"qm7_SLATM_global_data.npz", allow_pickle=True)
points=data["qm7_reps"]

chosen_indices=fps(points, subsetsize=2)
chosen_labels=data["qm7_labels"][chosen_indices]

print(chosen_indices)
print(chosen_labels)
np.save(outfolder+"indices.npy", chosen_indices)
np.save(outfolder+"labels.npy", chosen_labels)

#print_solution(points, chosen_indices, filename=outfolder+"fps")

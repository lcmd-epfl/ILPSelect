import numpy as np 
import timeit
import gurobipy as gp
from gurobipy import GRB
import pandas as pd

def addvariables(Z):
    upperbounds=[]
    I=[]
    J=[]
    for M in database_indices:
        CM=data["database_ncharges"][M]
        m=len(CM)
        I=I+[(i,j,M,G) for G in range(maxduplicates) for i in range(m) for j in range(n) if CM[i] == CT[j]] # if condition excludes j; i always takes all m values
        J=J+[(M,G) for G in range(maxduplicates)]

    x=Z.addVars(I, vtype=GRB.BINARY)
    y=Z.addVars(J, vtype=GRB.BINARY)
    print("Variables added.")
    return x,I,y

def addconstraints(Z,x,I,y):
    # bijection into [n]
    Z.addConstrs(x.sum('*',j,'*', '*') == 1 for j in range(n))
    
    for M in database_indices:
        CM=data["database_ncharges"][M]
        m=len(CM)
        # each i of each group is used at most once
        Z.addConstrs(x.sum(i,'*',M,G) <= 1 for i in range(m) for G in range(maxduplicates))
        # y[M,G] = OR gate of the x[i,j,M,G] for each (M,G) 
        Z.addConstrs(y[M,G] >= x[v] for G in range(maxduplicates) for v in I if v[2:]==(M,G))
        Z.addConstrs(y[M,G] <= x.sum('*','*',M,G) for G in range(maxduplicates))
    print("Constraints added.")
    return 0

# objective value should then be square rooted in the end (doesn't change optimality)
def setobjective(Z,x,I,y):
    print("Constructing objective function... ")
    key=0
    if(representation==0): # Coulomb case
        expr=gp.QuadExpr()
        T=targetdata['target_CMs'][target_index]
        for k in range(n):
            for l in range(n):
                expr += T[k,l]**2
        for M in database_indices:
            key=key+1
            Mol=data["database_CMs"][M]
            m=len(Mol)
            for G in range(maxduplicates):
                for (i,k) in [v[:2] for v in I if v[2:]==(M,G)]:
                    for (j,l) in [v[:2] for v in I if v[2:]==(M,G)]:
                        expr += (Mol[i,j]**2 - 2*T[k,l]*Mol[i,j])*x[i,k,M,G]*x[j,l,M,G]
                expr += y[M,G]*m*penaltyconst 
            print(key, "  /  ", size_database)
        expr=expr-n*penaltyconst

    else: # vector representations
        expr=gp.LinExpr()
        T=targetdata["target_reps"][target_index]
        for M in database_indices:
            key=key+1
            Mol=data["database_reps"][M]
            m=len(Mol)
            for G in range(maxduplicates):
                for (i,j) in [v[:2] for v in I if v[2:]==(M,G)]:
                    C=np.linalg.norm(Mol[i]-T[j])**2
                    #C=boundednorm(Mol[i]-T[j],2*penaltyconst) # need to experiment a bit more on that
                    if C==-1:
                        Z.addConstr(x[i,j,M,G]==0)
                    else:
                        expr += C*x[i,j,M,G]
                expr += y[M,G]*m*penaltyconst
            print(key, "  /  ", size_database)
        expr=expr-n*penaltyconst

    Z.setObjective(expr, GRB.MINIMIZE)
    print("Objective function set.")
    return 0

# computes L2 norm squared of v. If it exceeds k, return -1.
def boundednorm(v,k):
    norm=0
    for e in v:
        norm=norm+e**2
        if norm>k:
            return -1
    return norm

# Solution processing, output in "output_repname.csv".
def print_sols(Z, x, I, y):
    d={"SolN":[], "Fragments":[], "Excess":[], "ObjValNoPen":[], "ObjValWithPen":[], "Assignments":[]}
    SolCount=Z.SolCount
    print("Target has size", n)
    print("Using representation", repname)
    for solnb in range(SolCount):
        print()
        print("--------------------------------")
        Z.setParam("SolutionNumber",solnb)
        print("Processing solution number", solnb+1, "  /  ", SolCount)
        
        fragments=set()
        A=np.zeros((n,size_database,maxduplicates)) # A[j,M,G]
        for (i,j,M,G) in [v for v in I if np.rint(x[v].Xn)==1]:
            fragments.add((M,G))
            A[j,M,G]=i+1
        
        penalty=-n*penaltyconst
        amount_fragments=len(fragments)
        assignments=[]
        excess=[]
        fragmentlabels=[]
        k=0
        for (M,G) in fragments:
            used_indices=[]
            maps=[]
            m=len(data["database_ncharges"][M])
            penalty=penalty + m*penaltyconst
            fragmentlabels.append(data["database_labels"][M])
            for j in range(n):
                i=int(A[j,M,G]-1)
                if i>=0:
                    maps.append((i+1,j+1))
                    used_indices.append(i)
            assignments.append(maps)
            charges=np.array(data["database_ncharges"][M])
            excess.append(charges[np.delete(range(m),used_indices)].tolist())
            k=k+1
        d["Excess"].append(excess)
        d["Fragments"].append(fragmentlabels)
        d["SolN"].append(solnb+1)
        d["ObjValNoPen"].append(Z.PoolObjVal-penalty)
        d["ObjValWithPen"].append(Z.PoolObjVal)
        d["Assignments"].append(assignments)
             
    print(d)
    df=pd.DataFrame(d)
    print(df)
    print("Saving to output_"+repname+".csv.")
    df.to_csv("output_"+repname+".csv")
    return 0

def main():
    # construction of the model
    start=timeit.default_timer() 
    Z = gp.Model()
    Z.setParam('OutputFlag',1)
    x,I,y=addvariables(Z)
    addconstraints(Z,x,I,y)
    setobjective(Z,x,I,y)
    stop=timeit.default_timer()
    print("Model setup: ", stop-start, "s")
    
    # model parameters
    # PoolSearchMode 1/2 forces to fill the solution pool. 2 finds the best solutions.
    Z.setParam("PoolSearchMode", 1) 
    # these prevent non integral values although some solutions are still duplicating -- to fix?
    #Z.setParam("IntFeasTol", 1e-9)
    Z.setParam("IntegralityFocus", 1)
    Z.setParam("Method",2) # barrier method tends to keep integrality, reducing duplicate solutions. Method 1 is also possible for dual simplex.
    Z.setParam("NumericFocus",3) # computer should pay more attention to numerical errors at the cost of running time.
    Z.setParam("Quad",1) # should be redundant with Numeric Focus
    Z.setParam("MarkowitzTol",0.99) # should be redundant with Numeric Focus
    
    Z.setParam("TimeLimit", timelimit) 
    Z.setParam("PoolSolutions", numbersolutions)
    
    # optimization
    print("------------")
    print("Optimization")
    print("------------")
    Z.optimize()
    print("------------")
    print()
    print("Optimization runtime: ", Z.RunTime, "s")
    
    if(Z.status == 3):
        print("Model was proven to be infeasible.")
        return 1
    
    print_sols(Z,x,I,y)
    return 0

# modifiable global settings
target_index=0 # 0, 1, or 2 for qm9, vitc, or vitd.
maxduplicates=2 # number of possible copies of each molecule of the database
timelimit=3600# in seconds (not counting setup)
numbersolutions=50 # size of solution pool
representation=2 # 0 for Coulomb Matrix (CM), 1 for SLATM, 2 for aCM, 3 for SOAP, 4 for FCHL
penaltyconst=[1,1,10000,1,1][representation] # constant in front of size penalty

# global constants
repname=["CM", "SLATM", "aCM", "SOAP", "FCHL"][representation]
dataname="representations/database_"+repname+".npz"

data=np.load(dataname, allow_pickle=True)

targetdataname="representations/target_"+repname+"_data.npz"
targetdata=np.load(targetdataname, allow_pickle=True)
CT=targetdata['target_ncharges'][target_index]
n=len(CT)
targetname=["qm9", "vitc", "vitd"][target_index]

size_database=len(data["database_labels"]) # set this to a fixed number to take only first part of database
size_database=100
database_indices=range(size_database) 

main()

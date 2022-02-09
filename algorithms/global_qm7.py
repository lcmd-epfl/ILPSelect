import numpy as np 
from numpy import linalg
import timeit
import gurobipy as gp
from gurobipy import GRB
import pandas as pd
import sys

def addvariables(Z):
    I=[(M,G) for M in database_indices for G in range(maxduplicates)]
    x=Z.addVars(I, vtype=GRB.BINARY)
    print("Variables added.")
    return x

def addconstraints(Z,x):
    # sum of picked sizes bigger than size of target
    n=len(targetdata['target_ncharges'][target_index]) # size of target
    expr=gp.LinExpr() # number of atoms in picked molecules
    for M in database_indices:
        m=len(data["database_ncharges"][M]) # size of molecule M
        for G in range(maxduplicates):
            expr+=m*x[M,G]
    Z.addConstr(expr >= n)
    #Z.addConstr(expr <= 1.5*n)
    return 0

# objective value should then be square rooted in the end (doesn't change optimality)
def setobjective(Z,x):
    print("Constructing objective function... ")
    expr=gp.QuadExpr() # L2 squared distance from target rep to sum of chosen molecule reps
    penalty=gp.LinExpr()
    T=targetdata["target_reps"][target_index]

    penalty+=len(targetdata["target_ncharges"][target_index]) # number of atoms in target
    expr+=np.linalg.norm(T)**2
    for M in database_indices:
        print(M, "  /  ", size_database)
        for G in range(maxduplicates):
            CM=data["database_reps"][M]
            expr+=-2*T.T@CM * x[M,G]
            penalty -= len(data["database_ncharges"][M])*x[M,G] # number of atoms in M
            for MM in database_indices: 
                for GG in range(maxduplicates):
                    CMM=data["database_reps"][MM]
                    expr+=CM.T@CMM *x[M,G]*x[MM,GG]

    Z.setObjective(expr-penaltyconst*penalty, GRB.MINIMIZE)
    print("Objective function set.")
    return 0

# Solution processing, saved in "output_repname.csv".
def print_sols(Z, x):
    SolCount=Z.SolCount
    print("Using representation", repname)
    for solnb in range(SolCount):
        print()
        print("--------------------------------")
        print("Sol no", solnb)
        print("Objective value", Z.PoolObjVal)
        Z.setParam("SolutionNumber",solnb)
        for M in database_indices:
            for G in range(maxduplicates):
                if (np.rint(x[M,G].Xn)==1):
                    print(data["database_labels"][M])

        
    return 0

def main():
    # construction of the model
    start=timeit.default_timer() 
    Z = gp.Model()
    Z.setParam('OutputFlag',1)
    x=addvariables(Z)
    addconstraints(Z,x)
    setobjective(Z,x)
    stop=timeit.default_timer()
    print("Model setup: ", stop-start, "s")
    
    # model parameters
    # PoolSearchMode 1/2 forces to fill the solution pool. 2 finds the best solutions.
    Z.setParam("PoolSearchMode", 1) 
    # these prevent non integral values although some solutions are still duplicating -- to fix?
    Z.setParam("IntFeasTol", 1e-9)
    Z.setParam("IntegralityFocus", 1)
    Z.setParam("Method",1) # dual simplex method tends to keep integrality, reducing duplicate solutions. Method 0 is also possible for primal simplex.
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
    
    print_sols(Z,x)
    return 0

# modifiable global settings
target_index=0 # 0, 1, or 2 for qm9, vitc, or vitd.
maxduplicates=2 # number of possible copies of each molecule of the database
timelimit=3600# in seconds (not counting setup)
numbersolutions=5 # size of solution pool
representation=1 # 0 for SPAHM, 1 for CM

# global constants
repname=["SPAHM", "CM"][representation]
penaltyconst=[1e4,1][representation]

dataname="../representations/database_"+repname+"_global.npz"
data=np.load(dataname, allow_pickle=True)

targetdataname="../representations/target_"+repname+"_global_data.npz"
targetdata=np.load(targetdataname, allow_pickle=True)

targetname=["qm9", "vitc", "vitd"][target_index]

size_database=80#len(data["database_labels"]) # set this to a fixed number to take only first part of database
database_indices=range(size_database) 
main()

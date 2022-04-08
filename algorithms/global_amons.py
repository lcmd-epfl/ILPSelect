import numpy as np 
from numpy import linalg
import timeit
import gurobipy as gp
from gurobipy import GRB
import pandas as pd
import sys

def addvariables(Z):
    I=[(M,G) for M in database_indices for G in range(maxduplicates)] # indices of variable x
    x=Z.addVars(I, vtype=GRB.BINARY)
    y=Z.addVars(len(np.unique(targetdata["target_ncharges"][target_index])), vtype='I') # variable for each atom type in target
    print("Variables added.")
    return x,y

def addconstraints(Z,x,y):
    # constraints on x: sum of picked sizes bigger than size of target
    Tcharges = targetdata["target_ncharges"][target_index]
    n=len(Tcharges) # size of target
    expr=gp.LinExpr() # number of atoms in picked molecules
    for M in database_indices:
        m=len(data[targetname+"_amons_ncharges"][M]) # size of molecule M
        expr+=m*x.sum(M,'*')
    Z.addConstr(expr >= n)

    # constraints on y: 
    uniqueTcharges=np.unique(Tcharges, return_counts=True)
    penalties=[gp.LinExpr()+s for s in uniqueTcharges[1]]
    for M in database_indices:
        Mcharges=np.array(data[targetname+"_amons_ncharges"][M])
        for i in range(len(penalties)):
            penalties[i]-=np.count_nonzero(Mcharges==uniqueTcharges[0][i])*x.sum(M,'*')
    # need temporary variables to equate the penalty expression because otherwise gp.abs_ is confused and lost ;(
    temp=Z.addVars(len(penalties), vtype='I')
    Z.addConstrs(temp[i]==penalties[i] for i in range(len(penalties)))
    Z.addConstrs(y[i]==gp.abs_(temp[i]) for i in range(len(penalties)))
    return 0

# objective value is L2 square distance between target and sum of fragments plus some positive penalty
def setobjective(Z,x,y):
    print("Constructing objective function... ")
    expr=gp.QuadExpr() # L2 squared distance from target rep to sum of chosen molecule reps
    penalty=gp.LinExpr() # positive penalty added equal to sum over the atom types of max(0, number atoms in target - number of atoms in fragments)
    # this does not penalize picking an atom type that is not present in target -- but actually it implicitly does if we also penalize the size as before.
    T=targetdata["target_reps"][target_index]
   
    # penalty is excess number of atom (difference fragments - targe - target) + distances to fulfilling target atom types (y)
    penalty+=y.sum() 
    penalty-=len(targetdata["target_ncharges"][target_index]) # number of atoms in target
    expr+=np.linalg.norm(T)**2
    for M in database_indices:
        print(M, "  /  ", size_database)
        for G in range(maxduplicates):
            CM=data[targetname+"_amons_reps"][M]
            expr+=-2*T.T@CM * x[M,G]
            
            penalty += len(data[targetname+"_amons_ncharges"][M])*x[M,G] # number of atoms in M
            for MM in database_indices: 
                #print(MM, "  /  ", size_database)
                for GG in range(maxduplicates):
                    CMM=data[targetname+"_amons_reps"][MM]
                    expr+=CM.T@CMM *x[M,G]*x[MM,GG]

    Z.setObjective(expr+penaltyconst*penalty, GRB.MINIMIZE)
    print("Objective function set.")
    return 0

# Solution processing, saved in "output_repname.csv".
def print_sols(Z, x, y):
    d={"SolN":[], "Fragments":[], "ObjValNoPen":[], "ObjValWithPen":[]}
    SolCount=Z.SolCount
    print("Using representation", repname)
    for solnb in range(SolCount):
        Z.setParam("SolutionNumber",solnb)
        print()
        print("--------------------------------")
        print("Sol no", solnb)
        print("Objective value", Z.PoolObjVal)
        fragments=[]
        penalty=-len(targetdata["target_ncharges"][target_index]) # number of atoms in target
        for i in range(len(np.unique(targetdata["target_ncharges"][target_index]))):
            penalty+=y[i].Xn
        
        for M in database_indices:
            for G in range(maxduplicates):
                if (np.rint(x[M,G].Xn)==1):
                    print(data[targetname+"_amons_labels"][M])
                    fragments.append(data[targetname+"_amons_labels"][M])
                    penalty+=len(data[targetname+"_amons_ncharges"][M])
        
        d["SolN"].append(solnb+1)
        d["Fragments"].append(fragments)
        d["ObjValWithPen"].append(Z.PoolObjVal)
        d["ObjValNoPen"].append(Z.PoolObjVal-penalty*penaltyconst)
        
    df=pd.DataFrame(d)
    print(df)
    print("Saving to output_"+repname+"_global.csv")
    df.to_csv("output_"+repname+"_global.csv")
    return 0

def main():
    # construction of the model
    start=timeit.default_timer() 
    Z = gp.Model()
    Z.setParam('OutputFlag',1)
    x,y=addvariables(Z)
    addconstraints(Z,x,y)
    setobjective(Z,x,y)
    stop=timeit.default_timer()
    print("Model setup: ", stop-start, "s")
    
    # model parameters
    # PoolSearchMode 1/2 forces to fill the solution pool. 2 finds the best solutions.
    Z.setParam("PoolSearchMode", 2) 
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
    
    print_sols(Z,x,y)
    return 0

# modifiable global settings
target_index=0 # 0, 1, or 2 for qm9, vitc, or vitd.
maxduplicates=1 # number of possible copies of each molecule of the database
timelimit=43200# in seconds (not counting setup)
numbersolutions=20 # size of solution pool
representation=int(sys.argv[1])

# global constants
repname=["SLATM_2", "SLATM_3.5", "SLATM", "SLATM_8", "FCHL_2", "FCHL_3.5", "FCHL_4.8", "FCHL", "SOAP", "CM"][representation]
penaltyconst=[100, 100, 100, 100, 1, 1, 1, 1, 0.01, 1e4][representation]

dataname="../representations/amons_"+repname+"_global_data.npz"
data=np.load(dataname, allow_pickle=True)

targetdataname="../representations/target_"+repname+"_global_data.npz"
targetdata=np.load(targetdataname, allow_pickle=True)

targetname=["qm9", "vitc", "vitd"][target_index]

size_database=len(data[targetname+"_amons_labels"]) # set this to a fixed number to take only first part of database
database_indices=range(size_database) 
main()

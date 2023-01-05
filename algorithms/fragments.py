import numpy as np 
from numpy import linalg
import timeit
import time
import gurobipy as gp
from gurobipy import GRB
import pandas as pd
import sys

class model:
    """
    Molecule fragmentation algorithm to split given target with database elements using a common representation.
    ---------
    Expected file structures
    
    Database structure: npz file with keys 'labels', 'ncharges', and 'reps'.
    Target structure: npz file with keys 'ncharges' and 'rep'.
    See documentation.
    ---------
    Parameters of __init__
    
    path_to_database: string
        path to .npz file with database structure.
    path_to_target: string
        path to .npz file with target structure.
    ---------
    Parameters of setup
    
    scope: string
        defines shape of representations; takes value "local_matrix", "local_vector", or "global_vector".
    penalty_constant: int or float, optional (default=1e6)
        sets penalty constant in objective value.
    duplicates: int, optional (default=1)
        number of times the database is processed. Higher values than 1 replicate the database.
    ## move the two below to self.optimize
    nthreads: int, optional (default=0)
        number of threads used by Gurobi for the optimization. Lower values reduce the amount of memory used.
    poolgapabs: int or float, optional (default=GRB.INFINITY)
        absolute gap between best and worst solutions that are kept by Gurobi. This is used to filter extremely bad solutions.
    ---------
    Parameters of optimize
    
    number_of_solutions: int, optional (default=15)
        number of solutions computed by the algorithm.
    timelimit: int, optional (default=43200)
        number of seconds the optimization may run.
    poolsearchmode: int, optional (default=2)
        takes value 0, 1, or 2, and defines behavior of algorithm in terms of finding best solutions. See gurobi documentation.
    --------
    Parameter of output
    
    output_name: string, optional (default="../out/output.csv")
        path to a .csv file to write the solutions in.
    --------
    Attributes
    
    TODO
    --------
    Example

    >>> import fragments 
    >>>M=fragments.model("../representations/database.npz", "../representations/target.npz")
    >>>M.setup("global_vector",1e6)
    >>>M.optimize()
    >>>M.output()
    --------
    Reference
    [1] ******
    """

    ################### functions to call below ##################
    def __init__(self, path_to_database, path_to_target, scope, verbose=0):
        assert scope == "local_vector" or scope == "local_matrix" or scope == "global_vector", "Scope takes values local_matrix, local_vector, and global_vector only."
        self.database=np.load(path_to_database, allow_pickle=True)

        self.target=np.load(path_to_target, allow_pickle=True)

        self.size_database=len(self.database["labels"])
        #self.size_database=100 # uncomment this to only take first indices of the database for testing
        self.database_indices=range(self.size_database)
        self.scope=scope
        self.verbose=verbose
        self.temporaryconstraints=None
    
    def setup(self, penalty_constant=1e6, duplicates=1, nthreads=0, poolgapabs=GRB.INFINITY):
        # construction of the model
        self.duplicates=duplicates
        self.penalty_constant=penalty_constant

        start=timeit.default_timer() 

        self.Z = gp.Model()
        # model parameters
        self.Z.setParam('OutputFlag',self.verbose)

        self.Z.setParam("PreQLinearize", 0)
        self.Z.setParam("MIPFocus",1)
        self.Z.setParam("PoolGapAbs",poolgapabs)
        #### for memory issues in cluster
        #self.Z.setParam("Threads",nthreads) # decrease number of threads to prevent memory issues
        #self.Z.setParam("NodefileStart", 0.5)
        #self.Z.setParam("NodefileDir", "/scratch/haeberle/molekuehl")
        ####
        print("Parameters: penalty_constant=", penalty_constant, "; duplicates=", duplicates)

        self.x,self.y=self.addvariables(self.Z)
        self.addconstraints(self.Z,self.x,self.y)
        self.setobjective(self.Z,self.x,self.y)
        stop=timeit.default_timer()
        print("Model setup: ", stop-start, "s")
        return 0

    def optimize(self, number_of_solutions=15, timelimit=43200, PoolSearchMode=2):
        # optimization
        self.Z.setParam("PoolSearchMode", PoolSearchMode)
        self.Z.setParam("TimeLimit", timelimit) 
        self.Z.setParam("PoolSolutions", number_of_solutions)

        print("------------------------------------")
        print("           Optimization")
        print("Time limit: ", timelimit, " seconds")
        print("------------------------------------")
        self.Z.optimize()
        print()
        print("Optimization runtime: ", self.Z.RunTime, "s")
        if(self.Z.status == 3):
            print("Model was proven to be infeasible.")
            return 1
        return 0

    def output(self,output_name=None):
        self.print_sols(self.Z,self.x,self.y, output_name)
        return 0

    ############## tool functions below used by callable functions above ####################

    def addvariables(self,Z):
        print("Adding variables...")
        if(self.scope=="local_matrix" or self.scope=="local_vector"):
            Tcharges = self.target["ncharges"]
            n=len(Tcharges) # size of target
            upperbounds=[]
            self.I=[]
            J=[]
            for M in self.database_indices:
                Mcharges=self.database["ncharges"][M]
                m=len(Mcharges)
                self.I=self.I+[(i,j,M,G) for G in range(self.duplicates) for i in range(m) for j in range(n) if Mcharges[i] == Tcharges[j]] # if condition excludes j; i always takes all m values
                J=J+[(M,G) for G in range(self.duplicates)]

            x=Z.addVars(self.I, vtype=GRB.BINARY)
            y=Z.addVars(J, vtype=GRB.BINARY)
        elif(self.scope=="global_vector"):
            I=[(M,G) for M in self.database_indices for G in range(self.duplicates)] # indices of variable x
            x=Z.addVars(I, vtype=GRB.BINARY)
            #for M in [0,1,16,28,29,53,92]:
            #    x[M,0].start = 1
            y=Z.addVars(len(np.unique(self.target["ncharges"])), vtype='C') # variable for each atom type in target
        print("Variables added.")
        return x,y

    def addconstraints(self,Z,x,y):
        print("Adding constraints...")
        if(self.scope=="local_matrix" or self.scope=="local_vector"):
            n=len(self.target['ncharges']) # size of target
            # bijection into [n]
            Z.addConstrs(x.sum('*',j,'*', '*') == 1 for j in range(n))
                
            for M in self.database_indices:
                m=len(self.database['ncharges'][M])
                # each i of each group is used at most once
                Z.addConstrs(x.sum(i,'*',M,G) <= 1 for i in range(m) for G in range(self.duplicates))
                # y[M,G] = OR gate of the x[i,j,M,G] for each (M,G) 
                Z.addConstrs(y[M,G] >= x[v] for G in range(self.duplicates) for v in self.I if v[2:]==(M,G))
                Z.addConstrs(y[M,G] <= x.sum('*','*',M,G) for G in range(self.duplicates))

        elif(self.scope=="global_vector"):

            # constraints on x: sum of picked sizes bigger than size of target
            Tcharges = self.target["ncharges"]
            n=len(Tcharges) # size of target
            expr=gp.LinExpr() # number of atoms in picked molecules
            for M in self.database_indices:
                m=len(self.database["ncharges"][M]) # size of molecule M
                expr+=m*x.sum(M,'*')
            Z.addConstr(expr >= n)

            # constraints on y: 
            uniqueTcharges=np.unique(Tcharges, return_counts=True)
            penalties=[gp.LinExpr()+s for s in uniqueTcharges[1]]
            for M in self.database_indices:
                Mcharges=np.array(self.database["ncharges"][M])
                for i in range(len(penalties)):
                    penalties[i]-=np.count_nonzero(Mcharges==uniqueTcharges[0][i])*x.sum(M,'*')
            Z.addConstrs(y[i]>=penalties[i] for i in range(len(penalties)))
            Z.addConstrs(y[i]>=-penalties[i] for i in range(len(penalties)))
            print("Constraints added...")
            return 0 

    # objective value is L2 square distance between target and sum of fragments plus some positive penalty
    def setobjective(self,Z,x,y):
        print("Constructing objective function...")
        count=0
        if(self.scope=="local_matrix"): # Coulomb case
            expr=gp.QuadExpr()
            T=self.target['rep']
            n=len(self.target['ncharges']) # size of target
            for k in range(n):
                for l in range(n):
                    expr += T[k,l]**2
            for M in self.database_indices:
                count+=1
                print(count, "  /  ", self.size_database)
                Mol=self.database['reps'][M]
                m=len(Mol)
                for G in range(self.duplicates):
                    for (i,k) in [v[:2] for v in self.I if v[2:]==(M,G)]:
                        for (j,l) in [v[:2] for v in self.I if v[2:]==(M,G)]:
                            expr += (Mol[i,j]**2 - 2*T[k,l]*Mol[i,j])*x[i,k,M,G]*x[j,l,M,G]
                    expr += y[M,G]*m*self.penalty_constant
            expr=expr-n*self.penalty_constant
        elif(self.scope=="local_vector"):
            expr=gp.LinExpr()
            T=self.target['rep']
            n=len(self.target['ncharges']) # size of target
            for M in self.database_indices:
                count+=1
                print(count, "  /  ", self.size_database)
                Mol=self.database["reps"][M]
                m=len(Mol)
                for (i,j,G) in [(v[0],v[1],v[3]) for v in self.I if v[2]==M]:
                    C=np.linalg.norm(Mol[i]-T[j])**2
                    expr += C*x[i,j,M,G]
                if self.penalty_constant != 0:
                    expr += y.sum(M,'*')*m*self.penalty_constant
            expr=expr-n*self.penalty_constant

        elif(self.scope=="global_vector"):
            expr=gp.QuadExpr() # L2 squared distance from target rep to sum of chosen molecule reps
            penalty=gp.LinExpr() # positive penalty added equal to sum over the atom types of max(0, number atoms in target - number of atoms in fragments)
            # this does not penalize picking an atom type that is not present in target -- but actually it implicitly does if we also penalize the size as before.
            T=self.target["rep"]
           
            # penalty is excess number of atom (difference fragments - target) + distances to fulfilling target atom types (y)
            penalty+=y.sum() 
            penalty-=len(self.target["ncharges"]) # number of atoms in target
            expr+=np.linalg.norm(T)**2
            selfproducts=self.database['reps']@self.database['reps'].T
            targetproducts=self.database['reps']@self.target['rep']
            for M in self.database_indices:
                count+=1
                if self.verbose : print(count, "  /  ", self.size_database)
                for G in range(self.duplicates):
                    expr+=(selfproducts[M,M]-2*targetproducts[M]) * x[M,G]
                    if self.penalty_constant!=0:
                        penalty += len(self.database["ncharges"][M])*x[M,G] # number of atoms in M
                    for MM in [i for i in self.database_indices if i < M]: 
                        for GG in range(self.duplicates):
                            expr+=2*selfproducts[M,MM] *x[M,G]*x[MM,GG] # times two because of M and MM switch

            expr=expr+self.penalty_constant*penalty
        
        Z.setObjective(expr, GRB.MINIMIZE)
        print("Objective function set.")
        return 0

    # Solution processing, saved in "output_repname.csv".
    def print_sols(self,Z, x, y,output_name):
        self.SolCount=Z.SolCount
        if(self.scope=="local_matrix" or self.scope=="local_vector"):
            d={"SolN":[], "Fragments":[], "FragmentsID":[], "Excess":[], "ObjValNoPen":[], "ObjValWithPen":[], "Assignments":[]}
            n=len(self.target['ncharges']) # size of target
            SolCount=Z.SolCount
            for solnb in range(SolCount):
                Z.setParam("SolutionNumber",solnb)
                if self.verbose:
                    print()
                    print("--------------------------------")
                    print("Processing solution number", solnb+1, "  /  ", SolCount) 
                    print("Objective value", Z.PoolObjVal)
                fragments=set()
                # constructs matrix A with entry [j,M,G] that takes value i+1 if x[i,j,M,G]=1, and 0 otherwise
                A=np.zeros((n,self.size_database,self.duplicates)) 
                for (i,j,M,G) in [v for v in self.I if np.rint(x[v].Xn)==1]:
                    fragments.add((M,G))
                    A[j,M,G]=i+1
                
                penalty=-n
                amount_fragments=len(fragments)
                assignments=[]
                excess=[]
                fragmentlabels=[]
                fragmentsid=[]
                k=0
                for (M,G) in fragments:
                    used_indices=[]
                    maps=[]
                    m=len(self.database["ncharges"][M])
                    penalty+=m
                    fragmentlabels.append(self.database["labels"][M])
                    fragmentsid.append(M)
                    for j in range(n):
                        i=int(A[j,M,G]-1)
                        if i>=0:
                            maps.append((i+1,j+1))
                            used_indices.append(i)
                    assignments.append(maps)
                    charges=np.array(self.database["ncharges"][M])
                    excess.append(charges[np.delete(range(m),used_indices)].tolist())
                    k=k+1
                d["Excess"].append(excess)
                d["Fragments"].append(fragmentlabels)
                d["FragmentsID"].append(fragmentsid)
                d["SolN"].append(solnb+1)
                d["ObjValNoPen"].append(Z.PoolObjVal-penalty*self.penalty_constant)
                d["ObjValWithPen"].append(Z.PoolObjVal)
                d["Assignments"].append(assignments)
                     
            df=pd.DataFrame(d)
            df["Fragments"]=df["Fragments"].apply(lambda x:str(x))
            df=df.drop_duplicates(subset='Fragments')
            df=df.reset_index(drop=True)
        elif(self.scope=="global_vector"):
            d={"SolN":[], "Fragments":[], "FragmentsID":[], "ObjValNoPen":[], "ObjValWithPen":[]}
            SolCount=Z.SolCount
            for solnb in range(SolCount):
                Z.setParam("SolutionNumber",solnb)
                if self.verbose:
                    print()
                    print("--------------------------------")
                    print("Processing solution number", solnb+1, "  /  ", SolCount) 
                    print("Objective value", Z.PoolObjVal)
                fragments=[]
                fragmentsid=[]
                penalty=-len(self.target["ncharges"]) # number of atoms in target
                for i in range(len(np.unique(self.target["ncharges"]))):
                    penalty+=y[i].Xn
                
                for M in self.database_indices:
                    for G in range(self.duplicates):
                        if (np.rint(x[M,G].Xn)==1):
                            if self.verbose: print(self.database["labels"][M])
                            fragmentsid.append(M)
                            fragments.append(self.database["labels"][M])
                            penalty+=len(self.database["ncharges"][M])
                
                d["SolN"].append(solnb+1)
                d["Fragments"].append(fragments)
                d["FragmentsID"].append(fragmentsid)
                d["ObjValWithPen"].append(Z.PoolObjVal)
                d["ObjValNoPen"].append(Z.PoolObjVal-penalty*self.penalty_constant)
                df=pd.DataFrame(d)

        self.SolDict=d 
        if output_name != None:
            print("Saving to "+output_name+"...")
            df.to_csv(output_name)
            print("Saved.")
        return d
        
    def add_forbidden_combinations(self, fragmentsarray):
        for fragmentsid in fragmentsarray:
            expr=gp.LinExpr() 
            for M in fragmentsid:
                if(self.scope=="global_vector"):
                    expr+=self.x.sum(M,'*')
                else:
                    expr+=self.y.sum(M,'*')
            self.Z.addConstr(expr <= len(fragmentsid)-1)
        self.Z.update()
        return 0

    def randomsubset(self,p):
        if self.temporaryconstraints != None:
            self.Z.remove(self.temporaryconstraints)

        N=self.size_database
        mask=np.random.random_sample(N)<p
        keptindices=np.arange(0,N)[mask]
        lostindices=np.arange(0,N)[np.logical_not(mask)]
        
        if(self.scope=="global_vector"):
            c=self.Z.addConstrs(self.x.sum(M,'*') == 0 for M in lostindices)
        else:
            c=self.Z.addConstrs(self.y.sum(M,'*') == 0 for M in lostindices)

        self.temporaryconstraints=c
        return keptindices

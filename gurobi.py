import numpy as np 
import timeit
import gurobipy as gp
from gurobipy import GRB

# This function places the molecule M optimally within the target T such that 
# the charges of  the molecule CM and the charges of the target CT are
# compatible 

def OptimalPlacement(M,CM,T,CT):
        
    try:
        
        #Dimemsion of the  Molecule and Target
        m = len(M)
        n = len(T)         
        
        print("-----------------------------------")
        print("Sizes of Matrices to match: ", m, n)
        print("-----------------------------------")
        
        # Create a new model
        Z = gp.Model("QP")
        Z.setParam('OutputFlag', 0)
        start=timeit.default_timer()

        #Create Variables         
        x = Z.addVars(m, n, name='X', vtype=GRB.BINARY)        
        Z.addConstrs(x.sum(i, '*') == 1 for i in range(m))
        Z.addConstrs(x.sum('*',j) <= 1 for j in range(n))
        
        # if the charges of atom i in molecule and atom j in target do not 
        # match, then we forbid the assignment i-> j 
        for i in range(m):
           for j in range(n):
               if(CM[i] != CT[j]):
                   Z.addConstr(x[i,j] == 0)
        
        # Set objective by building a long expression 
        # this takes the most time by far, to optimize?
        expr = 2*x[1,1] * x[1,2]
        expr.clear()
        
        for i in range(m):
            for j in range(m):
                for k in range(n):
                    for l in range(n):
                        expr.add(x[i,k] * x[j,l], (T[k,l]-M[i,j])**2) 
                        
        expr.addConstant(-m)    
        Z.setObjective(expr, GRB.MINIMIZE)
        stop=timeit.default_timer()
        print("Time to define objective function:", stop-start, "s")
        # Optimize model
        print("\n----------------\n Optimisation... \n----------------\n")
        Z.optimize()
        assert Z.status == 2

        print("\n------------------\n Best match... \n------------------\n")
        matched_target = []
        for v in Z.getVars():
            if v.x == 1:
                i,j = v.varName.split("[")[-1].split("]")[0].split(",")
                j = int(j)
                matched_target.append(j)

        print("Indices of the target:")
        print(matched_target)
        print("Objective value:")
        print(Z.ObjVal)
        print("Runtime:")
        print(Z.Runtime, "s")

        return [matched_target, Z.ObjVal]
        
    except gp.GurobiError as e:
        print('Error code ' + str(e.errno) + ': ' + str(e))
        return [[], -1]
   
    except AttributeError:
        print('Encountered an attribute error')
        return [[],-1]
     
    except AssertionError:
        print("No optimal solution has been found")
        return [[],-1]

#A shortcut to call the OptimalPlacement function         
def Assignment(database_index, target_index):
    target_CM = data['target_CMs'][target_index]
    target_ncharges = data['target_ncharges'][target_index]
    target_label = data['target_labels'][target_index]
    assert len(target_CM) == len(target_ncharges)

    database_CM = data['database_CMs'][database_index]
    database_ncharges = data['database_ncharges'][database_index]
    database_label = data['database_labels'][database_index]
    assert len(database_CM) == len(database_ncharges)
    
    print("----------------------------------------")
    print("Trying to place", database_label, "inside", target_label, "...")
    print("----------------------------------------")
    [target_indices, objval] = OptimalPlacement(database_CM, database_ncharges, target_CM, target_ncharges)
    return [target_indices, objval]

if __name__ == "__main__":
    data = np.load("data.npz", allow_pickle=True)
    #fixed target index (value in {0, 1, 2})
    target_index = 2
    
    # loops over all candidates M, looking for minimal objective value
    size_database=len(data['database_ncharges'])
    m_index=0
    [target_indices, objval]=[[], -1]
    #for database_index in range(size_database):
    for database_index in [0,1]:
        print(database_index, "   /   ", size_database)
        [tempindices, tempobjval]=Assignment(database_index, target_index)
        if objval==-1 or (tempobjval!=-1 and tempobjval < objval):
            objval = tempobjval
            target_indices=tempindices
            molecule_index=database_index
     
    print("Best molecule M found:")
    print("Name:", data['database_labels'][m_index])
    print("Target indices:", target_indices)
    print("Objective value:", objval)


import gurobipy as gp
import numpy as np
from gurobipy import GRB
from scipy.spatial.distance import cdist


def add_variables(M, n):
    x = M.addVars(range(n), vtype="B")
    return x


def add_constraints(M, subsetsize, x):
    M.addConstr(x.sum() == subsetsize)
    return 0

def remove_variable(M, x, i):
    constr = M.addConstr(x[i]==0, name="TEMP")
    return constr

def reset_temp_constr(M, constr):
    if not constr is None: 
        M.remove(constr)
    return 0


def set_objective1(M, points, x):
    n = len(points)
    expr = gp.QuadExpr()
    for i in range(n):
        for j in range(i + 1, n):
            norm = np.linalg.norm(points[i] - points[j])
            expr += norm * x[i] * x[j]
    M.setObjective(expr, GRB.MAXIMIZE)
    return 0


def set_objective2(M, points, x):
    n = len(points)
    obj = M.addVar(vtype="C")
    print(1)
    maxnorm = np.max(cdist(points, points, metric='euclidean'))
    print(2)
    # maxnorm = 43
    for i in range(n):
        print(f"{i} / {n}")
        for j in range(i + 1, n):
            norm = np.linalg.norm(points[i] - points[j])
            M.addConstr(obj <= norm + maxnorm * (1 - x[i] + 1 - x[j]))
    M.setObjective(obj, GRB.MAXIMIZE)
    return 0


def read_solution(M, n, x):
    chosen_indices = []
    for i in range(n):
        print(i)
        if x[i].X == 1:
            chosen_indices.append(i)
    return np.array(chosen_indices)


def fps_subset(config):
    """
    Generate FPS subsets of size N for each target from the database.

    Parameters:
        parent_folder: absolute path of folder containing data/ folder with needed representations
        database: name of database (str) eg. "qm7"
        targets: array of names (array(str))
        representation: name of rep (str) eg. "FCHL"
        N: size of each subset
        in_database: whether the targets are in the database and should be removed from the ranking or not (bool)
    """

    parent_folder = config["current_folder"]
    representation = config["representation"]
    targets = config["target_names"]
    database = config["database"]
    N = config["learning_curve_ticks"][-1]
    in_database = config["in_database"]

    DATABASE_PATH = f"{parent_folder}data/{representation}_{database}.npz"
    database_info = np.load(DATABASE_PATH, allow_pickle=True)

    database_reps = database_info["reps"]
    if representation == "FCHL":
        database_global_rep = np.sum(database_reps, axis=1)
    else:
        print("Only FCHL is taken care of right now.")
        raise

    # n = number of points
    # d = dimension of space
    n, d = database_global_rep.shape
    assert N <= n, "Number of points is smaller than size of subset."

    M = gp.Model()

    x = add_variables(M, n)

    add_constraints(M, N, x)

    print("Constructing objective...")
    set_objective2(M, database_global_rep, x)
    print("Objective set.")

    M.update()
    
    # if not in database, need only N points
    if not in_database:
        M.optimize()
        assert M.status != 3, "Model is infeasible."

        ranking = read_solution(M, n, x)
        assert len(ranking)==N

        SAVE_PATH = f"{parent_folder}rankings/fps_{representation}_{database}.npy"
        np.save(SAVE_PATH, ranking)
        print(f"Saved FPS ranking of {N} fragments of database {database} to {SAVE_PATH}.")
        return 0

    # if in database, points need to be removed.
    # I don't take a subset of size N+1 and then remove, because FPS on N points is not a subset of FPS on N+1 points!
    constr = None
    for target_name in targets:
        reset_temp_constr(M, constr)
        M.update()

        target_index = np.where(database_info["labels"] == target_name)[0][0]
        if target_index <=1023:
            constr = remove_variable(M, x, target_index)

        M.optimize()
        assert M.status != 3, "Model is infeasible."

        ranking = read_solution(M, n, x)
        assert len(ranking)==N

        SAVE_PATH = f"{parent_folder}rankings/fps_{representation}_{database}_{target_name}.npy"
        np.save(SAVE_PATH, ranking)
        print(
            f"Saved FPS ranking of {N} fragments of database {database} without {target_name} to {SAVE_PATH}."
        )

    return 0

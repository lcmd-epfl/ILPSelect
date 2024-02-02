import gurobipy as gp
import numpy as np
from gurobipy import GRB
from scipy.spatial.distance import cdist


def add_variables(M, n):
    x = M.addVars(range(n), vtype="B")
    return x


def add_constraints(M, subsetsize, x):
    M.addConstr(x.sum() == subsetsize, name="SUBSETSIZE")
    return 0


def remove_variable(M, x, i):
    constr = M.addConstr(x[i] == 0, name="TEMP")
    return constr


def reset_temp_constr(M, constr):
    if not constr is None:
        M.remove(constr)
    return 0


def set_objective(M, points, x):
    n = len(points)
    obj = M.addVar(vtype="C")
    maxnorm = np.max(cdist(points, points, metric="euclidean"))
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
        if x[i].X == 1:
            chosen_indices.append(i)
    return np.array(chosen_indices)


def fps_subset(config):
    """
    Generate FPS subsets of each N of `learning_curve_ticks` for each target from the database.

    Parameters:
        config: TODO
    """

    parent_folder = config["current_folder"]
    representation = config["representation"]
    config_name = config["config_name"]
    targets = config["target_names"]
    database = config["database"]
    in_database = config["in_database"]
    N=config["learning_curve_ticks"]

    # we solve FPS horizontally (for each subset size for each target), store in a dictionary...
    # and regroup vertically (for each target for each subset size) and save.
    
    DATABASE_PATH = f"{parent_folder}data/{representation}_{database}_{config_name}.npz"
    database_info = np.load(DATABASE_PATH, allow_pickle=True)

    database_labels = database_info["labels"]
    database_reps = database_info["reps"]
    if representation == "FCHL":
        database_global_rep = np.sum(database_reps, axis=1)
    else:
        print("Only FCHL is taken care of right now.")
        raise

    # n = number of points
    n = len(database_global_rep)
    assert N[-1] <= n, "Number of points is smaller than size of subset."

    M = gp.Model()
    M.setParam("TimeLimit", config["timelimit"])

    x = add_variables(M, n)

    # Add constraint later, as it's subset size-dependent

    print("Constructing objective...")
    set_objective(M, database_global_rep, x)
    print("Objective set.")

    M.update()

    # all_rankings keys are (target_index, subset_size), with
    # target_index = -1 if it doesn't matter (if not in_database)
    all_rankings = {}

    for subset_size in N:

        # add size constraint
        size_constr = add_constraints(M, subset_size, x)

        if not in_database:
            ranking = fps_ranking(config, subset_size, database_labels, M, x)[0]

            all_rankings[(-1, subset_size)] = ranking

        else:
            rankings = fps_ranking(config, subset_size, database_labels, M, x)
            for i in range(len(targets)):
                ranking = rankings[i]

                all_rankings[(i,subset_size)]=ranking

        # remove size constraint for later
        M.remove(size_constr)

    # merge all rankings of each target inside a single array and save
                
    # not target-dependent
    if not in_database:
        rankings = []
        for subset_size in N:
            rankings.append(all_rankings[(-1,subset_size)])
        
        SAVE_PATH = f"{parent_folder}rankings/fps_{representation}_{database}.npy"
        np.save(SAVE_PATH, rankings)
        print(
            f"Saved FPS rankings of {N} fragments of database {database} to {SAVE_PATH}."
        )

    # target-dependent
    for i in range(len(targets)):
        target_name = targets[i]
        rankings = []
        for subset_size in range(N):
            rankings.append(all_rankings[(i,subset_size)])

        SAVE_PATH = (
            f"{parent_folder}rankings/fps_{representation}_{database}_{target_name}.npy"
        )
        np.save(SAVE_PATH, ranking)
        print(
            f"Saved FPS ranking of {subset_size} fragments of database {database} without {target_name} to {SAVE_PATH}."
        )

    return 0

def fps_ranking(config, size_subset, database_labels, M, x):
    """
    Generate FPS subsets of size size_subset for each target from the database.
    This disregards the entry `learning_curve_ticks` of `config`.

    Parameters:
        config: TODO
        size_subset: size of subset (int)
        database_labels: database (npz)
        M: gurobi FPS model to solve (gurobipy.Model)
        x: variables of model M

    Returns:
        rankings (array of array)
    """

    targets = config["target_names"]
    in_database = config["in_database"]

    n=len(database_labels)

    # if not in database, need only N points
    if not in_database:
        M.optimize()
        assert M.status != 3, "Model is infeasible."

        ranking = read_solution(M, n, x)
        assert len(ranking) == size_subset, "Ranking output size does not match input condition."

        return [ranking]

    # if in database, points need to be removed.
    # I don't take a subset of size N+1 and then remove, because FPS on N points is not a subset of FPS on N+1 points!
   
    constr = None
    ranking = None
    rankings = []
    for target_name in targets:
        reset_temp_constr(M, constr)
        M.update()

        target_index = np.where(database_labels == target_name)[0][0]

        # if the ranking includes the target, recompute!
        if (ranking is None) or (target_index in ranking):
            constr = remove_variable(M, x, target_index)

            M.optimize()
            assert M.status != 3, "Model is infeasible."

            ranking = read_solution(M, n, x)
            assert len(ranking) == size_subset, "Ranking output size does not match input condition."
        else:
            constr = None

        rankings.append(ranking)

    return rankings
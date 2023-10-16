import timeit

import gurobipy as gp
import numpy as np
import pandas as pd
from gurobipy import GRB


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
    ## TODO move the two below to self.optimize
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
    Parameters of output

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
    # TODO: move initial arguments to the setup phase?
    # since it's not useful when reading from file (it's used right now but may be changed)
    def __init__(self, path_to_database, path_to_target, scope, verbose=False):
        assert scope in [
            "local_vector",
            "local_matrix",
            "global_vector",
        ], "Scope takes values local_matrix, local_vector, and global_vector only."
        self.database = np.load(path_to_database, allow_pickle=True)
        self.database_reps = self.database["reps"]
        self.database_ncharges = self.database["ncharges"]
        self.database_labels = self.database["labels"]

        self.target = np.load(path_to_target, allow_pickle=True)
        self.target_rep = self.target["rep"]
        self.target_ncharges = self.target["ncharges"]

        self.size_database = len(self.database_labels)
        # self.size_database=20 # uncomment this to only take first indices of the database for testing
        self.database_indices = range(self.size_database)
        self.scope = scope
        self.verbose = verbose
        self.temporaryconstraints = None
        self.visitedfragments = []
        self.solutions = {"Fragments": [], "Value": []}
        self.objbound = None
        self.number_of_fragments = None

    def setup(self, penalty_constant=1, duplicates=1):
        # construction of the model
        self.duplicates = duplicates
        self.penalty_constant = float(penalty_constant)

        start = timeit.default_timer()

        self.Z = gp.Model()

        # model parameters
        # useless now?
        # self.Z.setParam("PreQLinearize", 0)
        # self.Z.setParam("MIPFocus",1)

        print(
            "Parameters: penalty_constant=",
            penalty_constant,
            "; duplicates=",
            duplicates,
        )

        self.x, self.y = self.addvariables(self.Z)
        self.addconstraints(self.Z, self.x, self.y)
        self.setobjective(self.Z, self.x, self.y)
        stop = timeit.default_timer()
        print("Model setup: ", stop - start, "s")
        return 0

    def optimize(
        self,
        number_of_solutions=20,
        timelimit=600,
        PoolSearchMode=1,
        poolgapabs=GRB.INFINITY,
        nthreads=0,
        callback=False,
        objbound=40,
        number_of_fragments=20,
    ):
        # model parameters
        self.Z.setParam("OutputFlag", self.verbose)
        self.Z.setParam("PoolSearchMode", PoolSearchMode)
        self.Z.setParam("TimeLimit", timelimit)
        self.Z.setParam("PoolSolutions", number_of_solutions)

        self.Z.setParam("PoolGapAbs", poolgapabs)
        #### for memory issues in cluster
        self.Z.setParam(
            "Threads", nthreads
        )  # decrease number of threads to decrease memory use
        # self.Z.setParam("NodefileStart", 0.5)
        # self.Z.setParam("NodefileDir", "/scratch/haeberle/molekuehl")
        ####

        print("------------------------------------")
        print("           Optimization")
        print("Time limit: ", timelimit, " seconds")
        print("------------------------------------")
        if callback:
            self.Z.setParam("LazyConstraints", 1)
            self.objbound = objbound
            self.number_of_fragments = number_of_fragments
            self.Z.optimize(
                lambda _, where: self.callback(where)
            )  # argument model is already in self
        else:
            self.Z.optimize()
        print()
        print("Optimization runtime: ", self.Z.RunTime, "s")
        if self.Z.status == 3:
            print("Model was proven to be infeasible.")
            return 1
        return 0

    def readmodel(self, filepath):
        self.Z = gp.read(filepath)

        # finding duplicates and x, y variables
        self.duplicates = 0

        x = gp.tupledict()
        y = gp.tupledict()
        Varlist = self.Z.getVars()
        for v in Varlist:
            vname = v.VarName
            if vname[0] == "x":
                x.update({(eval(vname[2:-1])): v})
            elif vname[0] == "y":
                y.update({(eval(vname[2:-1])): v})
                if vname[2] == "0":
                    self.duplicates += 1
        self.x = x
        self.y = y

        # finding penalty constant
        Mcharges = self.database_ncharges[0]
        m = len(Mcharges)
        # z = indicator variable for fragments
        if self.scope == "global_vector":
            z = x
        else:
            z = y

        self.penalty_constant = z[0, 0].obj / m
        print(f"Found penalty constant of {self.penalty_constant}.")
        return 0

    def changepenalty(self, newpenalty):
        self.penalty_constant = newpenalty
        Mcharges = self.database_ncharges[0]
        m = len(Mcharges)

        # z = indicator variable for fragments
        if self.scope == "global_vector":
            z = self.x
        else:
            z = self.y

        penratio = newpenalty / z[0, 0].obj * m  # =newpen/oldpen
        for i in z.keys():
            obj = z[i].obj
            z[i].setAttr("obj", obj * penratio)

        self.Z.setAttr("ObjCon", self.Z.ObjCon * penratio)
        self.Z.update()
        print("New penalty:", newpenalty)
        return 0

    # example filepath '../out/model.mps'
    def savemodel(self, filepath):
        if self.temporaryconstraints != None:
            self.Z.remove(self.temporaryconstraints)
        self.Z.write(filepath)
        return 0

    def output(self, output_name=None):
        d = self.print_sols(self.Z, self.x, self.y, output_name)
        return d

    ############## functions for subset selection ###############

    def add_forbidden_combinations(self, fragmentsarray):
        for fragmentsid in fragmentsarray:
            expr = gp.LinExpr()
            self.add_visited_fragments(fragmentsid)
            for M in fragmentsid:
                if self.scope == "global_vector":
                    expr += self.x.sum(M, "*")
                else:
                    expr += self.y.sum(M, "*")
            self.Z.addConstr(expr <= len(fragmentsid) - 1)
        self.Z.update()
        return 0

    def remove_fragments(self, fragmentsarray):
        # removes every fragment of a solution, that is, an array of array of fragment ids
        for fragmentsid in fragmentsarray:
            for M in fragmentsid:
                if self.scope == "global_vector":
                    self.Z.addConstr(self.x.sum(M, "*") == 0)
                else:
                    self.Z.addConstr(self.y.sum(M, "*") == 0)
        self.Z.update()
        return 0

    def remove_fragment_name(self, fragment_name):
        # remove lone fragment by name
        fragment_id = np.where(self.database_labels == fragment_name)[0][0]

        self.remove_fragments([[fragment_id]])
        return 0

    def randomsubset(self, p):
        if self.temporaryconstraints != None:
            self.Z.remove(self.temporaryconstraints)

        N = self.size_database
        mask = np.random.random_sample(N) < p
        keptindices = np.arange(0, N)[mask]
        lostindices = np.arange(0, N)[np.logical_not(mask)]

        if self.scope == "global_vector":
            c = self.Z.addConstrs(self.x.sum(M, "*") == 0 for M in lostindices)
        else:
            c = self.Z.addConstrs(self.y.sum(M, "*") == 0 for M in lostindices)

        self.temporaryconstraints = c
        return keptindices

    def add_cps_constraint(self):
        # z = indicator variable for fragments
        if self.scope == "global_vector":
            z = self.x
        else:
            z = self.y
        self.Z.addConstr(z.sum() == 1)
        # remove bijection constraints
        for C in self.Z.getConstrs():
            if C.sense == "=":
                self.Z.remove(C)
        self.Z.update()
        return 0

    ############## tool functions below used by callable functions above ####################

    # adds solutions and objective value to dictionary self.solutions
    # adds fragment index to self.visitedfragments if not already inside
    def add_visited_fragments(self, frags):
        for M in frags:
            # if not np.any(np.isin(self.visitedfragments, M)):
            if not M in self.visitedfragments:
                self.visitedfragments.append(M)
        return 0

    # only used by self.callback() !
    def add_to_solutions(self, frags):
        self.add_visited_fragments(frags)
        self.solutions["Fragments"].append(frags)
        self.solutions["Value"].append(self.Z.cbGet(GRB.Callback.MIPSOL_OBJ))
        return 0

    # used only by self.callback() !
    #
    def add_lazy_constraint(self):
        # var is the variable indicator of fragments (x or y)
        if self.scope == "global_vector":
            var = self.x
        else:
            var = self.y

        # values of var, 1 if fragment is picked, 0 otherwise.
        frags = self.Z.cbGetSolution(var)
        S = []
        for i in frags.keys():
            if np.abs(frags[i] - 1) < 1e-5:
                # expr+=var[i]
                S.append(i[0])

        # adds found combination with objective value to solutions and visitedfragments
        self.add_to_solutions(S)
        # forces new fragment to appear: expr sums over indices NOT in visitedfragments.
        # expr=gp.LinExpr()
        # self.Z.cbLazy(expr <= len(S)-1) # forbids combination found
        # I=[i for (i,_) in var.keys() if not np.any(np.isin(self.visitedfragments, i))]
        I = [i for (i, _) in var.keys() if not i in self.visitedfragments]
        expr = var.sum(I, "*")
        self.Z.cbLazy(expr >= 1)

        return 0

    # argument model is already in self
    def callback(self, where):
        # adds lazy constraint whenever a solution is found that is within poolgapabs bounds
        if (where == GRB.Callback.MIPSOL) and self.Z.cbGet(
            GRB.Callback.MIPSOL_OBJ
        ) < self.objbound:
            if self.verbose:
                print(self.Z.cbGet(GRB.Callback.MIPSOL_OBJ))
            self.add_lazy_constraint()
            if self.verbose:
                print(len(self.visitedfragments))
            if len(self.visitedfragments) >= self.number_of_fragments:
                self.Z.terminate()
                print("Interrupting because enough fragments were found.")
        return 0

    def addvariables(self, Z):
        print("Adding variables...")
        if self.scope == "local_matrix" or self.scope == "local_vector":
            Tcharges = self.target_ncharges
            Trep = self.target_rep
            n = len(Tcharges)  # size of target
            upperbounds = []
            I = []
            J = []
            for M in self.database_indices:
                Mcharges = self.database_ncharges[M]
                Mrep = self.database_reps[M]
                m = len(Mcharges)
                I = I + [
                    (i, j, M, G)
                    for G in range(self.duplicates)
                    for i in range(m)
                    for j in range(n)
                    if Mcharges[i] == Tcharges[j]
                    and np.linalg.norm(Mrep[i] - Trep[j]) < 1  # EXPERIMENTAL
                ]  # if condition excludes j; i always takes all m values
                J = J + [(M, G) for G in range(self.duplicates)]

            y = Z.addVars(J, vtype=GRB.BINARY, name="y")
            x = Z.addVars(I, vtype=GRB.BINARY, name="x")
        elif self.scope == "global_vector":
            I = [
                (M, G) for M in self.database_indices for G in range(self.duplicates)
            ]  # indices of variable x
            x = Z.addVars(I, vtype=GRB.BINARY, name="x")
            # for M in [0,1,16,28,29,53,92]:
            #    x[M,0].start = 1
            y = Z.addVars(
                len(np.unique(self.target_ncharges)), vtype="C", name="y"
            )  # variable for each atom type in target
        print("Variables added.")
        return x, y

    def addconstraints(self, Z, x, y):
        print("Adding constraints...")
        if self.scope == "local_matrix" or self.scope == "local_vector":
            n = len(self.target_ncharges)  # size of target
            # bijection into [n]
            Z.addConstrs((x.sum("*", j, "*", "*") == 1 for j in range(n)), name="bij")
            I = x.keys()

            for M in self.database_indices:
                m = len(self.database_ncharges[M])
                # each i of each group is used at most once #TODO: does this make sense?
                Z.addConstrs(
                    x.sum(i, "*", M, G) <= 1
                    for i in range(m)
                    for G in range(self.duplicates)
                )
                # y[M,G] = OR gate of the x[i,j,M,G] for each (M,G)
                Z.addConstrs(
                    y[M, G] >= x[v]
                    for G in range(self.duplicates)
                    for v in I
                    if v[2:] == (M, G)
                )
                Z.addConstrs(
                    y[M, G] <= x.sum("*", "*", M, G) for G in range(self.duplicates)
                )  # not needed if y is pushed down?

        elif self.scope == "global_vector":
            # constraints on x: sum of picked sizes bigger than size of target
            Tcharges = self.target_ncharges
            n = len(Tcharges)  # size of target
            expr = gp.LinExpr()  # number of atoms in picked molecules
            for M in self.database_indices:
                m = len(self.database_ncharges[M])  # size of molecule M
                expr += m * x.sum(M, "*")
            Z.addConstr(expr >= n)

            # constraints on y:
            uniqueTcharges = np.unique(Tcharges, return_counts=True)
            penalties = [gp.LinExpr() + s for s in uniqueTcharges[1]]
            for M in self.database_indices:
                Mcharges = np.array(self.database_ncharges[M])
                for i in range(len(penalties)):
                    penalties[i] -= np.count_nonzero(
                        Mcharges == uniqueTcharges[0][i]
                    ) * x.sum(M, "*")
            Z.addConstrs(y[i] >= penalties[i] for i in range(len(penalties)))
            Z.addConstrs(y[i] >= -penalties[i] for i in range(len(penalties)))
            print("Constraints added...")
            return 0

    # objective value is L2 square distance between target and sum of fragments plus some positive penalty
    def setobjective(self, Z, x, y):
        print("Constructing objective function...")
        count = 0
        if self.scope == "local_matrix":  # Coulomb case
            expr = gp.QuadExpr()
            I = x.keys()
            T = self.target_rep
            n = len(self.target_ncharges)  # size of target
            for k in range(n):
                for l in range(n):
                    expr += T[k, l] ** 2
            for M in self.database_indices:
                count += 1
                if self.verbose:
                    print(count, "  /  ", self.size_database)
                Mcharges = self.database_ncharges[M]
                Mol = self.database_reps[M]
                m = len(Mcharges)
                for G in range(self.duplicates):
                    for i, k in [v[:2] for v in I if v[2:] == (M, G)]:
                        for j, l in [v[:2] for v in I if v[2:] == (M, G)]:
                            expr += (
                                (Mol[i, j] ** 2 - 2 * T[k, l] * Mol[i, j])
                                * x[i, k, M, G]
                                * x[j, l, M, G]
                            )
                    expr += y[M, G] * m * self.penalty_constant
            expr = expr - n * self.penalty_constant

        elif self.scope == "local_vector":
            expr = gp.LinExpr()
            I = x.keys()
            T = self.target_rep
            n = len(self.target_ncharges)  # size of target
            for M in self.database_indices:
                count += 1
                if self.verbose:
                    print(count, "  /  ", self.size_database)
                Mcharges = self.database_ncharges[M]
                Mol = self.database_reps[M]
                m = len(Mcharges)
                for i, j, G in [(v[0], v[1], v[3]) for v in I if v[2] == M]:
                    C = np.linalg.norm(Mol[i] - T[j]) ** 2
                    expr += C * x[i, j, M, G]
                if self.penalty_constant != 0:
                    expr += y.sum(M, "*") * m * self.penalty_constant
            expr = expr - n * self.penalty_constant

        elif self.scope == "global_vector":
            expr = (
                gp.QuadExpr()
            )  # L2 squared distance from target rep to sum of chosen molecule reps
            penalty = (
                gp.LinExpr()
            )  # positive penalty added equal to sum over the atom types of max(0, number atoms in target - number of atoms in fragments)
            # this does not penalize picking an atom type that is not present in target -- but actually it implicitly does if we also penalize the size as before.
            T = self.target_rep

            # penalty is excess number of atom (difference fragments - target) + distances to fulfilling target atom types (y)
            penalty += y.sum()
            penalty -= len(self.target_ncharges)  # number of atoms in target
            expr += np.linalg.norm(T) ** 2
            selfproducts = self.database_reps @ self.database_reps.T
            targetproducts = self.database_reps @ self.target_rep
            for M in self.database_indices:
                count += 1
                if self.verbose:
                    print(count, "  /  ", self.size_database)
                for G in range(self.duplicates):
                    expr += (selfproducts[M, M] - 2 * targetproducts[M]) * x[M, G]
                    if self.penalty_constant != 0:
                        penalty += (
                            len(self.database_ncharges[M]) * x[M, G]
                        )  # number of atoms in M
                    for MM in [i for i in self.database_indices if i < M]:
                        for GG in range(self.duplicates):
                            expr += (
                                2 * selfproducts[M, MM] * x[M, G] * x[MM, GG]
                            )  # times two because of M and MM switch

            expr = expr + self.penalty_constant * penalty

        Z.setObjective(expr, GRB.MINIMIZE)
        print("Objective function set.")
        return 0

    # Solution processing, saved in "output_repname.csv".
    def print_sols(self, Z, x, y, output_name):
        self.SolCount = Z.SolCount
        if self.scope == "local_matrix" or self.scope == "local_vector":
            d = {
                "SolN": [],
                "Fragments": [],
                "FragmentsID": [],
                "Excess": [],
                "ObjValNoPen": [],
                "ObjValWithPen": [],
                "Assignments": [],
            }
            n = len(self.target_ncharges)  # size of target
            SolCount = Z.SolCount
            I = x.keys()
            for solnb in range(SolCount):
                Z.setParam("SolutionNumber", solnb)
                if self.verbose:
                    print()
                    print("--------------------------------")
                    print("Processing solution number", solnb + 1, "  /  ", SolCount)
                    print("Objective value", Z.PoolObjVal)
                fragments = set()
                # constructs matrix A with entry [j,M,G] that takes value i+1 if x[i,j,M,G]=1, and 0 otherwise
                A = np.zeros((n, self.size_database, self.duplicates))
                for i, j, M, G in [v for v in I if np.rint(x[v].Xn) == 1]:
                    fragments.add((M, G))
                    A[j, M, G] = i + 1

                penalty = -n
                amount_fragments = len(fragments)
                assignments = []
                excess = []
                fragmentlabels = []
                fragmentsid = []
                k = 0
                for M, G in fragments:
                    used_indices = []
                    maps = []
                    m = len(self.database_ncharges[M])
                    penalty += m
                    fragmentlabels.append(self.database_labels[M])
                    fragmentsid.append(M)
                    for j in range(n):
                        i = int(A[j, M, G] - 1)
                        if i >= 0:
                            maps.append((i + 1, j + 1))
                            used_indices.append(i)
                    assignments.append(maps)
                    charges = np.array(self.database_ncharges[M])
                    excess.append(charges[np.delete(range(m), used_indices)].tolist())
                    k = k + 1
                d["Excess"].append(excess)
                d["Fragments"].append(fragmentlabels)
                d["FragmentsID"].append(fragmentsid)
                d["SolN"].append(solnb + 1)
                d["ObjValNoPen"].append(Z.PoolObjVal - penalty * self.penalty_constant)
                d["ObjValWithPen"].append(Z.PoolObjVal)
                d["Assignments"].append(assignments)

            df = pd.DataFrame(d)
            df["Fragments"] = df["Fragments"].apply(lambda x: str(x))
            df = df.drop_duplicates(subset="Fragments")
            df = df.reset_index(drop=True)
        elif self.scope == "global_vector":
            d = {
                "SolN": [],
                "Fragments": [],
                "FragmentsID": [],
                "ObjValNoPen": [],
                "ObjValWithPen": [],
            }
            SolCount = Z.SolCount
            for solnb in range(SolCount):
                Z.setParam("SolutionNumber", solnb)
                if self.verbose:
                    print()
                    print("--------------------------------")
                    print("Processing solution number", solnb + 1, "  /  ", SolCount)
                    print("Objective value", Z.PoolObjVal)
                fragments = []
                fragmentsid = []
                penalty = -len(self.target_ncharges)  # number of atoms in target
                for i in range(len(np.unique(self.target_ncharges))):
                    penalty += y[i].Xn

                for M in self.database_indices:
                    for G in range(self.duplicates):
                        if np.rint(x[M, G].Xn) == 1:
                            if self.verbose:
                                print(self.database_labels[M])
                            fragmentsid.append(M)
                            fragments.append(self.database_labels[M])
                            penalty += len(self.database_ncharges[M])

                d["SolN"].append(solnb + 1)
                d["Fragments"].append(fragments)
                d["FragmentsID"].append(fragmentsid)
                d["ObjValWithPen"].append(Z.PoolObjVal)
                d["ObjValNoPen"].append(Z.PoolObjVal - penalty * self.penalty_constant)
                df = pd.DataFrame(d)

        self.SolDict = d
        if output_name != None:
            print("Saving to " + output_name + "...")
            df.to_csv(output_name)
            print("Saved.")
        return d

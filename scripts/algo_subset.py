import numpy as np
import pandas as pd
from scripts import fragments


def algo_subset(config):
    """
    Generates size N subset of database indices from a .mps file (Gurobi model). 
    The corresponding model must already have been generated using `scripts.algo_model.algo_model`.

    Parameters:
        config: config dictionary. Must contain keys `"repository_folder"`, `"penalty"`, `"representation"`, `"target_names"`,
            `"database"`, `"learning_curve_ticks"`, `"config_name"`, `"scope"`, `"verbose"`, `"PoolSearchMode"`, `"timelimit"`.
    """

    repository_path = config["repository_folder"]
    pen = config["penalty"]
    representation = config["representation"]
    targets = config["target_names"]
    database = config["database"]
    N = config["learning_curve_ticks"][-1]
    config_name=config["config_name"]

    DATA_PATH = f"{repository_path}data/{representation}_{database}_{config_name}.npz"

    for target_name in targets:
        TARGET_PATH = (
            f"{repository_path}data/{representation}_{target_name}.npz"
        )
        M = fragments.model(
            DATA_PATH, TARGET_PATH, scope=config["scope"], verbose=config["verbose"]
        )

        MODEL_PATH = f"{repository_path}models/{representation}_{database}_{target_name}_{pen}.mps"
        try:
            M.readmodel(MODEL_PATH)
        except:
            print(f"WARNING: model of target {target_name} not in models/ folder. Path name must be {MODEL_PATH}.")
            continue

        # reads already found combinations to remove then (if we want to continue previous optimization for example)
        # df=pd.read_csv(outfolder+"newsolutions"+str(pen)+".csv")
        # M.add_forbidden_combinations(df['Fragments'].apply(eval))
        if config["in_database"]:
            M.remove_fragment_name(target_name)

        # optimize with callback
        M.optimize(
            number_of_solutions=100,
            PoolSearchMode=config["PoolSearchMode"],
            timelimit=config["timelimit"],
            poolgapabs=100,
            callback=True,
            objbound=100,
            number_of_fragments=N,
        )

        solution_df = pd.DataFrame(M.solutions)
        SOLUTION_SAVE_PATH = f"{repository_path}solutions/{representation}_{database}_{target_name}_{pen}.csv"
        solution_df.to_csv(SOLUTION_SAVE_PATH)

        print(f"Saved {len(solution_df)} solutions to {SOLUTION_SAVE_PATH}.")

        # sorts solutions found by objective value
        # and saves fragments to file
        sorteddf = solution_df.sort_values("Value")["Fragments"]
        ordered_frags = []
        for e in sorteddf:
            for f in e:
                if not f in ordered_frags:
                    ordered_frags.append(f)

        RANKING_SAVE_PATH = f"{repository_path}rankings/algo_{representation}_{database}_{target_name}_{pen}.npy"
        np.save(RANKING_SAVE_PATH, ordered_frags)

        print(f"Saved ranking of size {len(ordered_frags)} to {RANKING_SAVE_PATH}.")

    return 0

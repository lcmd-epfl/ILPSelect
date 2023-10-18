config = {
    # absolute paths
    "current_folder": "/home/haeberle/molekuehl/cluster/",
    "repository_folder": "/home/haeberle/molekuehl/",
    ###
    "config_name": "qm7fragments",
    "database": "qm7",
    "representation": "FCHL",
    ###
    # filled in below
    "target_names": [],
    "in_database": True,
    "plot_average_target_names": [],
    ###
    "generate_database": False,
    "generate_targets": False,
    "cur_subset": False,
    "fps_subset": False,  # FPS not implemented yet
    "sml_subset": False,
    "algo_model": False,
    "algo_subset": False,
    "learning_curves": [], #["fragments", "sml", "cur", "random"],
    "plots_individual": ["algo", "sml", "cur", "random"],
    "plots_average": ["algo", "sml", "cur", "random"],
    ###
    "scope": "local_vector",
    "penalty": 1,
    "duplicates": 1,
    "timelimit": 1 * 3600,  # 1 hours
    "PoolSearchMode": 2,
    "number_of_fragments": 1024,  # size of subset selected
    "verbose": False,
    ###
    "learning_curve_ticks": [2**k for k in range(4, 11)],
    ###
    "random_state": None,  # for multiple random subset selection, don't use a fixed state!
    "CV": 5,  # number of cross-validation for random learning curves
}

############ for 10 targets inside qm7 ############
import numpy as np
import pandas as pd

qm7 = np.load("/home/haeberle/molekuehl/cluster/data/FCHL_qm7.npz", allow_pickle=True)
qm7_df = pd.DataFrame({"ncharges": qm7["ncharges"], "labels": qm7["labels"]})

num_heavy_atoms = qm7_df["ncharges"].map(lambda charges: sum(charges != 1))

# take 10 with fixed random state (doesn't matter so much)
target_sample = qm7_df[num_heavy_atoms >= 7]["labels"].sample(10, random_state=42).values

config["target_names"] = target_sample
config["plot_average_target_names"] = target_sample
config["in_database"] = True

config = {
    # absolute paths
    "current_folder": "/home/haeberle/molekuehl/cluster/",
    "repository_folder": "/home/haeberle/molekuehl/",
    ###
    "config_name": "qm9qm9",
    "database": "qm9",
    "representation": "FCHL",
    ###
    # filled in below
    "target_names": [],
    "in_database": True,
    "plot_average_target_names": [],
    ###
    "generate_database": True,
    "generate_targets": True,
    "cur_subset": True,
    "fps_subset": False, # FPS seg fault on qm9 for now.. 
    "sml_subset": True,
    "algo_model": True,
    "algo_subset": True,
    "learning_curves": ["fragments", "sml", "cur", "random"],
    "plots_individual": ["algo", "sml", "cur", "random"],
    "plots_average": ["algo", "sml", "cur", "random"],
    ###
    "scope": "local_vector",
    "penalty": 0,
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

############ for 10 targets inside qm9 ############
import numpy as np
import pandas as pd

DATA_PATH = f"{config['current_folder']}data/qm9_data.npz"
qm9 = np.load(DATA_PATH, allow_pickle=True)
qm9_df = pd.DataFrame({"ncharges": qm9["ncharges"]}).reset_index()

num_heavy_atoms = qm9_df["ncharges"].map(lambda charges: sum(charges != 1))

# take 10 with fixed random state (for reproducibility)
target_sample = qm9_df[num_heavy_atoms >= 7]["index"].sample(10, random_state=42).values

config["target_names"] = target_sample
config["plot_average_target_names"] = target_sample
config["in_database"] = True

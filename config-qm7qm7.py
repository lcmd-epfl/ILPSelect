import os

config = {
    # absolute paths
    "repository_folder": os.path.dirname(__file__)+'/',
    #"repository_folder": "/home/haeberle/Documents/molekuehl/",
    ###
    "config_name": "qm7qm7",
    "database": "qm7",
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
    "fps_subset": True,
    "sml_subset": True,
    "algo_model": True,
    "algo_subset": True,
    "learning_curves": ["algo", "fps", "sml", "cur", "random"],
    "plots_individual": ["algo", "fps", "sml", "cur", "random"],
    "plots_average": ["algo", "fps", "sml", "cur", "random"],
    ###
    "scope": "local_vector",
    "penalty": 1,
    "duplicates": 1,
    "timelimit": 10 * 3600,  # 10 hours
    "PoolSearchMode": 2,
    "number_of_fragments": 1024,  # size of subset selected
    "verbose": True,
    ###
    "learning_curve_ticks": [2**k for k in range(4, 11)],
    "FPS_timelimit": 600, # 10 mins
    ###
    "random_state": None,  # for multiple random subset selection, don't use a fixed state!
    "CV": 5,  # number of cross-validation for random learning curves
}

############ for 10 targets inside qm7 ############
import numpy as np
import pandas as pd
DATA_PATH = f"{config['repository_folder']}data/{config['representation']}_{config['database']}_{config['config_name']}.npz"

try:
    qm7 = np.load(DATA_PATH, allow_pickle=True)
except:
    print("Generating database in order to pick out random targets.")
    from scripts.generate import generate_database
    generate_database(config)
    config["generate_database"]=False
    qm7 = np.load(DATA_PATH, allow_pickle=True)

qm7_df = pd.DataFrame({"ncharges": qm7["ncharges"], "labels": qm7["labels"]})

num_heavy_atoms = qm7_df["ncharges"].map(lambda charges: sum(charges != 1))

# take 10 with fixed random state (doesn't matter so much)
target_sample = (
    qm7_df[num_heavy_atoms >= 7]["labels"].sample(10, random_state=42).values
)

config["target_names"] = target_sample
config["plot_average_target_names"] = target_sample
config["in_database"] = True

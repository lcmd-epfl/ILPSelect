"""
This is an example config file which finished in 1h30 on the JED cluster (EPFL).
Sildenafil is the only target. The learning curves are done only for subset sizes N=16, 32, 64, with no cross-validation for the random curve.

See `config-qm7drugs.py` for a full config file.
"""

config = {
    # absolute path
    "repository_folder": "/home/haeberle/molekuehl/",
    ###
    "config_name": "template",
    "database": "qm7",
    "representation": "FCHL",
    ###
    # corresponding names must be in targets/targets.csv
    "target_names": [
        "sildenafil",
    ],
    "in_database": False,
    "plot_average_target_names": [
        "sildenafil",
    ],
    ###
    "generate_database": True,
    "generate_targets": True,
    "cur_subset": True,
    "fps_subset": True,
    "sml_subset": True,
    "algo_model": True,
    "algo_subset": True,
    "learning_curves": ["algo", "sml", "cur", "random"],
    "plots_individual": ["algo", "sml", "cur", "random"],
    "plots_average": ["algo", "sml", "cur", "random"],
    ###
    "scope": "local_vector",
    "penalty": 1,
    "penalty_lc": [0,1],
    "duplicates": 1,
    "timelimit": 3600,  # 1 hour
    "PoolSearchMode": 2,
    "number_of_fragments": 64,  # size of subset selected by ILP
    ###
    "FPS_timelimit": 600, # 10 mins
    "verbose": False,
    ###
    "learning_curve_ticks": [16, 32, 64],
    ###
    "CV": 1,  # number of cross-validation for random learning curves
}

"""
This is an example config file which should compile in __ minutes. 
Sildenafil is the only target. The learning curves are done only for subset sizes N=16, 32, 64, with no cross-validation for the random curve.

See `config-qm7drugs.py` for a full config file.
"""

config = {
    # absolute paths
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
    "generate_database": False,
    "generate_targets": False,
    "cur_subset": False,
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
    "duplicates": 1,
    "timelimit": 1 * 3600,  # 1 hours
    "PoolSearchMode": 2,
    "number_of_fragments": 64,  # size of subset selected
    "verbose": False,
    ###
    "learning_curve_ticks": [16, 32, 64],
    ###
    "random_state": None,  # for multiple random subset selection, don't use a fixed state!
    "CV": 1,  # number of cross-validation for random learning curves
}

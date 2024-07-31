config = {
    # absolute paths
    "repository_folder": "/home/haeberle/molekuehl/",
    ###
    "config_name": "qm7drugs",
    "database": "qm7",
    "representation": "FCHL",
    ###
    # corresponding names must be in targets/targets.csv
    "target_names": [
        "sildenafil",
        "penicillin",
        "troglitazone",
        "imatinib",
        "pemetrexed",
        "oxycodone",
        "pregabalin",
        "apixaban",
        "salbutamol",
        "oseltamivir",
    ],
    "in_database": False,
    "plot_average_target_names": [
        "sildenafil",
        "penicillin",
        "troglitazone",
        "imatinib",
        "pemetrexed",
        "oxycodone",
        "pregabalin",
        "apixaban",
        "salbutamol",
        "oseltamivir",
    ],
    ###
    "generate_database": True,
    "generate_targets": True,
    "cur_subset": False,
    "fps_subset": False,
    "sml_subset": False,
    "algo_model": False,
    "algo_subset": False,
    "learning_curves": ["algo", "sml", "fps", "cur", "random"],
    "plots_individual": ["algo", "fps", "sml", "cur", "random"],
    "plots_average": ["algo", "fps", "cur", "random"],
    ###
    "scope": "local_vector",
    "penalty": 0,
    "duplicates": 1,
    "timelimit": 1 * 3600,  # 1 hours
    "PoolSearchMode": 2,
    "number_of_fragments": 1024,  # size of subset selected
    "verbose": True,
    ###
    "FPS_timelimit": 600, # 10 mins
    "learning_curve_ticks": [2**k for k in range(4, 11)],
    ###
    "random_state": None,  # for multiple random subset selection, don't use a fixed state!
    "CV": 5,  # number of cross-validation for random learning curves
}

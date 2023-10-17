config = {
    # absolute paths
    "current_folder": "/home/haeberle/molekuehl/cluster/",
    "repository_folder": "/home/haeberle/molekuehl/",
    ###
    "config_name": "drugs",
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
    "cur_subset": True,
    "fps_subset": False,  # FPS not implemented yet
    "sml_subset": True,
    "algo_model": True,
    "algo_subset": True,
    "learning_curves": ["fragments", "sml", "cur", "random"],
    "plots_individual": ["fragments", "sml", "cur", "random"],
    "plots_average": ["fragments", "sml", "cur", "random"],
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


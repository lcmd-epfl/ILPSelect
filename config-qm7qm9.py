config = {
    # absolute paths
    "repository_folder": "/home/haeberle/molekuehl/",
    ###
    "config_name": "qm7qm9",
    "database": "qm7",
    "representation": "FCHL",
    ###
    # corresponding names must be in targets/targets.csv
    "target_names": [
        "121259",
        "12351",
        "35811",
        "85759",
        "96295",
        "5696",
        "31476",
        "55607",
        "68076",
        "120425",
    ],
    "in_database": False,
    "plot_average_target_names": [
        "121259",
        "12351",
        "35811",
        "85759",
        "96295",
        "5696",
        "31476",
        "55607",
        "68076",
        "120425",
    ],
    ###
    "generate_database": True,
    "generate_targets": True,
    "cur_subset": True,
    "fps_subset": True,
    "sml_subset": True,
    "algo_model": True,
    "algo_subset": True,
    "learning_curves": ["algo", "sml", "fps", "cur", "random"],
    "plots_individual": ["algo", "fps", "sml", "cur", "random"],
    "plots_average": ["algo", "fps", "sml", "cur", "random"],
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
    "FPS_timelimit": 600, # 10 mins
    ###
    "random_state": None,  # for multiple random subset selection, don't use a fixed state!
    "CV": 5,  # number of cross-validation for random learning curves
}

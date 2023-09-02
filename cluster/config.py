config = {
    # absolute paths
    "current_folder": "/home/haeberle/molekuehl/cluster/",
    "repository_folder": "/home/haeberle/molekuehl/",
    ###
    "database": "qm7",
    "remove_target_from_database": False,
    "representation": "FCHL",
    ###
    "target_names": [
        "penicillin",
        "troglitazone",
    ],  # corresponding names must be in targets/targets.csv
    ###
    "generate_database": False,
    "generate_targets": True,
    "sml_subset": True,
    "algo_model": True,
    "algo_subset": True,
    "learning_curves": True,
    "learning_curves_random": True,
    "plots": True,
    ###
    "scope": "local_vector",
    "penalty": 0,
    "duplicates": 1,
    "timelimit": 4 * 3600,  # 4 hours
    "PoolSearchMode": 2,
    "number_of_fragments": 1024,  # size of subset selected
    "verbose": False,
    ###
    # "learning_curve_ticks": [2**k for k in range(4, 11)],
    "learning_curve_ticks": [2**k for k in range(4, 5)],
    ###
    "random_state": None,  # for multiple random subset selection, don't use a fixed state!
}

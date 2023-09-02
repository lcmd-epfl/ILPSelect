config = {
    # absolute paths
    "current_folder": "/home/haeberle/molekuehl/cluster/",
    "repository_folder": "/home/haeberle/molekuehl/",
    ###
    "database": "qm7",
    "remove_target_from_database": False,
    "representation": "FCHL",
    ###
    "target_names": ["sildenafil"],  # corresponding names must be in targets/targets.csv
    ###
    "generate_database": False,
    "generate_targets": False,
    "sml_subset": False,
    "algo_model": False,
    "algo_subset": False,
    "learning_curves": False,
    "learning_curves_random": False,
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
    "learning_curve_ticks": [2**k for k in range(4, 11)],
    ###
    "random_state": None,  # for multiple random subset selection, don't use a fixed state!
}

import os

config = {
    # absolute paths
    "repository_folder": os.path.dirname(__file__)+'/',
    #"repository_folder": "/home/haeberle/molekuehl/",
    ###
    "config_name": "qm7drugs",
    "database": "qm7",
    "representation": "FCHL",
    ###
    # corresponding names must be in targets/energies.csv (targets/all_targets.csv)
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
    "penalty_lc": [0,1],
    "duplicates": 1,
    "timelimit": 10 * 3600,  # 10 hours
    "PoolSearchMode": 2,
    "number_of_fragments": 1024,  # size of subset selected
    "verbose": True,
    ###
    "learning_curve_ticks": [2**k for k in range(4, 11)],
    "FPS_timelimit": 600, # 10 mins
    ###
    "CV": 5,  # number of cross-validation for random learning curves
}

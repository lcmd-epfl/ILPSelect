import os

number_of_scan_points = 11

config = {
    # absolute paths
    "repository_folder": os.path.dirname(__file__)+'/',
    ###
    "config_name": "qm7imatinibscan",
    "database": "qm7",
    "representation": "FCHL",
    ###
    # corresponding names must be in targets/energies.csv (targets/all_targets.csv)
    "target_names": [f"imatinib_{i:03d}" for i in range(1,number_of_scan_points)],
    "in_database": False,
    "plot_average_target_names": [f"imatinib_{i:03d}" for i in range(1,number_of_scan_points)],
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
    "plots_average": ["algo", "fps", "sml", "cur", "random"],
    ###
    "scope": "local_vector",
    "penalty": 0,
    "penalty_lc": [0, 1],
    "duplicates": 1,
    "timelimit": 10 * 3600,  # 10 hours
    "PoolSearchMode": 2,
    "number_of_fragments": 1024,  # size of subset selected
    "verbose": True,
    ###
    "learning_curve_ticks": [2**k for k in range(4, 11)],
    #"learning_curve_ticks": [4**k for k in range(2, 6)],
    #"learning_curve_ticks": [16, 1024],
    "FPS_timelimit": 600, # 10 mins
    ###
    "CV": 5,  # number of cross-validation for random learning curves
}

from algorithms import fragments


def algo_model(config):
    """
    Generates ILP model to a .mps file in models/ folder.
    Warning: the data of the database and targets must already be generated in the data/ folder!

    Parameters:
        repository_path: absolute path to repository
        database: database name (str) eg. "qm7"
        targets: array of target names (array(str))
        representation: name of representation (str) eg. "FCHL"
        config: config dictionary. Must contain keys "penalty" and "scope".
    """

    repository_path = config["repository_folder"]
    pen = config["penalty"]
    representation = config["representation"]
    targets = config["target_names"]
    database = config["database"]

    for target_name in targets:
        DATA_PATH = f"{repository_path}cluster/data/{representation}_{database}.npz"
        TARGET_PATH = f"{repository_path}cluster/data/{representation}_{target_name}.npz"
        M = fragments.model(
            DATA_PATH, TARGET_PATH, scope=config["scope"], verbose=config["verbose"]
        )

        # sets up model and writes to a file. Only needed once.
        M.setup(penalty_constant=pen, duplicates=1)

        SAVE_PATH = (
            f"{repository_path}cluster/models/{representation}_{database}_{target_name}_{pen}.mps"
        )

        M.savemodel(SAVE_PATH)

        print(
            f"Generated model of representation {representation} for database {database} and target {target_name} with penalty {pen}."
        )

    return 0

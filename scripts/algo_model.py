from scripts import fragments
import os.path

def algo_model(config):
    """
    Generates ILP model to a .mps file in models/ folder.
    Warning: the data of the database and targets must already be generated in the data/ folder!

    Parameters:
        repository_path: absolute path to repository
        database: database name (str) eg. "qm7"
        targets: array of target names (array(str))
        representation: name of representation (str) eg. "FCHL"
        config: config dictionary. Must contain keys `"repository_folder"`, `"penalty"`, `"representation"`, `"target_names"`,
            `"database"`, `"config_name"`, `"scope"`, `"verbose"`.
    """

    repository_path = config["repository_folder"]
    pen = config["penalty"]
    representation = config["representation"]
    targets = config["target_names"]
    database = config["database"]
    config_name=config["config_name"]

    for target_name in targets:


        DATA_PATH = f"{repository_path}data/{representation}_{database}_{config_name}.npz"
        TARGET_PATH = (
            f"{repository_path}data/{representation}_{target_name}.npz"
        )
        M = fragments.model(
            DATA_PATH, TARGET_PATH, scope=config["scope"], verbose=config["verbose"]
        )

        SAVE_PATH = f"{repository_path}models/{representation}_{database}_{target_name}_{pen}.mps"

        # if pen=1 file already exists, open it, change the penalty, and save.
        PEN_1_PATH = f"{repository_path}models/{representation}_{database}_{target_name}_1.mps"
        if os.path.isfile(PEN_1_PATH):
            M.readmodel(PEN_1_PATH)
            M.changepenalty(pen)
            M.savemodel(SAVE_PATH)
            continue

        # otherwise, sets up model from scratch and writes to a file. Only needed once.
        M.setup(penalty_constant=pen, duplicates=1)

        M.savemodel(SAVE_PATH)

        print(f"Saved model to file {SAVE_PATH}.")

    return 0

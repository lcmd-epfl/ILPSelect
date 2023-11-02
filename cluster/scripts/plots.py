import pickle

import numpy as np
import pandas as pd
import plotly.graph_objects as go

Ha2kcal = 627.5


def target_energy(config, target_name):
    repository_path = config["repository_folder"]
    representation = config["representation"]
    database = config["database"]

    with open(f"{repository_path}cluster/data/atom_energy_coeffs.pickle", "rb") as f:
        atom_energy_coeffs = pickle.load(f)

    TARGET_PATH = f"{repository_path}cluster/data/{representation}_{target_name}.npz"

    target_info = np.load(TARGET_PATH, allow_pickle=True)
    Q_target = target_info["ncharges"]

    if config["in_database"]:
        # label of target
        Y_PATH = f"{repository_path}{database}/energies.csv"
        y_target = pd.read_csv(Y_PATH).query("file == @target_name")["energy / Ha"].iloc[0]

        # removing target from database
        mask = database_labels != target_name
        X = X[mask]
        Q = Q[mask]
        database_labels = database_labels[mask]

    else:
        Y_PATH = f"{repository_path}cluster/targets/energies.csv"
        y_target = pd.read_csv(Y_PATH).query("file == @target_name+'.xyz'")["energy / Ha"].iloc[0]

    # REMOVED AS PER RUBEN'S COMMENTS
    # # y energies offset
    # for ncharge in Q_target:
    #     y_target -= atom_energy_coeffs[ncharge]

    return y_target


def plots_individual(config):
    """
    Draw combined plots of the learning curves for each target and saves them to plots/.

    Parameters:
        config: config dictionary. Must contain keys "learning_curve_ticks", "random_state", "in_database" TODO
    """

    learning_curve_ticks = config["learning_curve_ticks"]
    parent_directory = config["current_folder"]
    repository_path = config["repository_folder"]
    pen = config["penalty"]
    representation = config["representation"]
    targets = config["target_names"]
    database = config["database"]
    curves = config["plots_individual"]

    PERCENTAGE_ERROR = True

    # individual plots
    for target_name in targets:
        # normalization constant is energy of target
        normalization = 1
        if PERCENTAGE_ERROR:
            y_target = target_energy(config, target_name)
            normalization = np.abs(y_target * Ha2kcal) / 100

        print("Normalizing constant:", normalization)

        # create figure and axis
        N = learning_curve_ticks

        fig = go.Figure()

        if "algo" in curves:
            ALGO_CURVE_PATH = f"{parent_directory}learning_curves/algo_{representation}_{database}_{target_name}_{pen}.npz"
            ALGO_CURVE = np.load(ALGO_CURVE_PATH, allow_pickle=True)
            ALGO = ALGO_CURVE["mae"] * Ha2kcal / normalization
            # plot learning curve ALGO
            fig.add_trace(go.Scatter(x=N, y=ALGO, name="Algorithm"))

        if "sml" in curves:
            SML_CURVE_PATH = f"{parent_directory}learning_curves/sml_{representation}_{database}_{target_name}.npz"
            SML_LEARNING_CURVE = np.load(SML_CURVE_PATH, allow_pickle=True)
            SML = SML_LEARNING_CURVE["mae"] * Ha2kcal / normalization
            # plot learning curve SML
            fig.add_trace(go.Scatter(x=N, y=SML, name="SML"))

        if "fps" in curves:
            FPS_CURVE_PATH = f"{parent_directory}learning_curves/fps_{representation}_{database}_{target_name}.npz"

            FPS_LEARNING_CURVE = np.load(FPS_CURVE_PATH, allow_pickle=True)
            FPS = FPS_LEARNING_CURVE["mae"] * Ha2kcal / normalization
            # plot learning curve FPS
            fig.add_trace(go.Scatter(x=N, y=FPS, name="FPS"))

        if "cur" in curves:
            CUR_CURVE_PATH = f"{parent_directory}learning_curves/cur_{representation}_{database}_{target_name}.npz"

            CUR_LEARNING_CURVE = np.load(CUR_CURVE_PATH, allow_pickle=True)
            CUR = CUR_LEARNING_CURVE["mae"] * Ha2kcal / normalization
            # plot learning curve CUR
            fig.add_trace(go.Scatter(x=N, y=CUR, name="CUR"))

        if "random" in curves:
            RANDOM_CURVE_PATH = f"{parent_directory}learning_curves/random_{representation}_{database}_{target_name}.npz"
            RANDOM_CURVE = np.load(RANDOM_CURVE_PATH, allow_pickle=True)
            MEAN_RANDOM, STD_RANDOM = (
                np.mean(RANDOM_CURVE["all_maes_random"], axis=0) * Ha2kcal / normalization,
                np.std(RANDOM_CURVE["all_maes_random"], axis=0) * Ha2kcal / normalization,
            )

            # plot learning curve random with std as error bars
            CV = len(RANDOM_CURVE["all_maes_random"])

            fig.add_trace(
                go.Scatter(
                    x=N,
                    y=MEAN_RANDOM,
                    error_y=dict(
                        type="data",  # value of error bar given in data coordinates
                        array=STD_RANDOM,
                        visible=True,
                    ),
                    name=f"Random ({CV}-fold)",
                )
            )

        fig.update_layout(
            yaxis=dict(type="log"),
            xaxis=dict(tickmode="array", tickvals=N, type="log"),
            xaxis_title="Training set size",
            yaxis_title="MAE [kcal/mol]",
            title=f"Learning curve on target {target_name}",
        )

        if PERCENTAGE_ERROR:
            fig.update_layout(yaxis=dict(type="linear"), yaxis_title="MAPE [%]")

        SAVE_PATH = f"{parent_directory}plots/{representation}_{database}_{target_name}_{pen}.png"
        fig.write_image(SAVE_PATH)
        print(f"Saved plot to {SAVE_PATH}")

        # fig.show()

    return 0


def plots_average(config):
    """
    Draw average plots of the learning curves and saves them to plots/.

    Parameters:
        config: TODO
    """

    N = config["learning_curve_ticks"]
    parent_directory = config["current_folder"]
    pen = config["penalty"]
    representation = config["representation"]
    targets = config["plot_average_target_names"]
    database = config["database"]
    curves = config["plots_average"]

    fig = go.Figure()

    PERCENTAGE_ERROR = True
    if "algo" in curves:
        ALGO = []
        for target_name in targets:
            # normalization constant is energy of targett
            normalization = 1
            if PERCENTAGE_ERROR:
                y_target = target_energy(config, target_name)
                normalization = np.abs(y_target * Ha2kcal) / 100

            ALGO_CURVE_PATH = f"{parent_directory}learning_curves/algo_{representation}_{database}_{target_name}_{pen}.npz"
            ALGO_CURVE = np.load(ALGO_CURVE_PATH, allow_pickle=True)
            ALGO.append(ALGO_CURVE["mae"] * Ha2kcal / normalization)
        ALGO = np.mean(ALGO, axis=0)

        fig.add_trace(go.Scatter(x=N, y=ALGO, name="Average algorithm"))

    if "sml" in curves:
        SML = []
        for target_name in targets:
            # normalization constant is energy of targett
            normalization = 1
            if PERCENTAGE_ERROR:
                y_target = target_energy(config, target_name)
                normalization = np.abs(y_target * Ha2kcal) / 100

            SML_CURVE_PATH = f"{parent_directory}learning_curves/sml_{representation}_{database}_{target_name}.npz"
            SML_LEARNING_CURVE = np.load(SML_CURVE_PATH, allow_pickle=True)
            SML.append(SML_LEARNING_CURVE["mae"] * Ha2kcal / normalization)
        SML = np.mean(SML, axis=0)

        fig.add_trace(go.Scatter(x=N, y=SML, name="Average SML"))

    if "fps" in curves:
        FPS = []
        for target_name in targets:
            # normalization constant is energy of target
            normalization = 1
            if PERCENTAGE_ERROR:
                y_target = target_energy(config, target_name)
                normalization = np.abs(y_target * Ha2kcal) / 100

            FPS_CURVE_PATH = f"{parent_directory}learning_curves/fps_{representation}_{database}_{target_name}.npz"
            FPS_LEARNING_CURVE = np.load(FPS_CURVE_PATH, allow_pickle=True)
            FPS.append(FPS_LEARNING_CURVE["mae"] * Ha2kcal / normalization)
        FPS = np.mean(FPS, axis=0)

        fig.add_trace(go.Scatter(x=N, y=FPS, name="Average FPS"))

    if "cur" in curves:
        CUR = []
        for target_name in targets:
            # normalization constant is energy of targett
            normalization = 1
            if PERCENTAGE_ERROR:
                y_target = target_energy(config, target_name)
                normalization = np.abs(y_target * Ha2kcal) / 100

            CUR_CURVE_PATH = f"{parent_directory}learning_curves/cur_{representation}_{database}_{target_name}.npz"
            CUR_LEARNING_CURVE = np.load(CUR_CURVE_PATH, allow_pickle=True)
            CUR.append(CUR_LEARNING_CURVE["mae"] * Ha2kcal / normalization)
        CUR = np.mean(CUR, axis=0)

        fig.add_trace(go.Scatter(x=N, y=CUR, name="Average CUR"))

    if "random" in curves:
        MEAN_RANDOM = []
        STD_RANDOM = []
        for target_name in targets:
            # normalization constant is energy of targett
            normalization = 1
            if PERCENTAGE_ERROR:
                y_target = target_energy(config, target_name)
                normalization = np.abs(y_target * Ha2kcal) / 100

            RANDOM_CURVE_PATH = f"{parent_directory}learning_curves/random_{representation}_{database}_{target_name}.npz"
            RANDOM_CURVE = np.load(RANDOM_CURVE_PATH, allow_pickle=True)
            MEAN_RANDOM.append(
                np.mean(RANDOM_CURVE["all_maes_random"], axis=0) * Ha2kcal / normalization
            )
            STD_RANDOM.append(
                np.std(RANDOM_CURVE["all_maes_random"], axis=0) * Ha2kcal / normalization
            )
        MEAN_RANDOM = np.mean(MEAN_RANDOM, axis=0)
        # square std to get variance, sum because of independence, and sqrt again to get back std.
        # Var((X + Y)/2) = (Var(X) + Var(Y))/4 for X, Y independent (no covariance)
        # STD_RANDOM = np.mean(STD_RANDOM, axis=0)
        STD_RANDOM = np.sqrt(np.sum(np.array(STD_RANDOM) ** 2, axis=0)) / len(STD_RANDOM)

        fig.add_trace(
            go.Scatter(
                x=N,
                y=MEAN_RANDOM,
                error_y=dict(
                    type="data",  # value of error bar given in data coordinates
                    array=STD_RANDOM,
                    visible=True,
                ),
                name="Average random",
            )
        )

    fig.update_layout(
        yaxis=dict(type="log"),
        xaxis=dict(tickmode="array", tickvals=N, type="log"),
        xaxis_title="Training set size",
        yaxis_title="MAE [kcal/mol]",
        title=f"Average learning curves on {len(targets)} targets",
    )

    if PERCENTAGE_ERROR:
        fig.update_layout(yaxis=dict(type="linear"), yaxis_title="MAPE [%]")

    SAVE_PATH = f"{parent_directory}plots/{representation}_{database}_average_{pen}.png"
    fig.write_image(SAVE_PATH)
    print(f"Saved plot to {SAVE_PATH}")

    # fig.show()
    return 0

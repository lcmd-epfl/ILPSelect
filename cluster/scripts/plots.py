import numpy as np
import plotly.graph_objects as go

Ha2kcal = 627.5


def plots_individual(
    parent_directory,
    database,
    targets,
    representation,
    pen,
    learning_curve_ticks,
    curves=["algo", "sml", "cur", "random"],
    #curves=["algo", "sml", "fps", "cur", "random"],
):
    """
    Draw combined plots of the learning curves for each target and saves them to plots/.

    Parameters:
        parent_directory: cluster/ absolute path
        database: name of database (str) eg "qm7"
        targets: array of target names (array(str))
        representation: representation (str) eg "FCHL"
        pen: penalty of the solutions for the file names (int, float or str)
        curves: list of curves to compute plots for
        in_database: whether the targets are in the database or not (bool). CUR and FPS rankings have different file names if this is the case.
    """

    # individual plots
    for target_name in targets:
        # create figure and axis
        N = learning_curve_ticks

        fig = go.Figure()

        if "algo" in curves:
            ALGO_CURVE_PATH = f"{parent_directory}learning_curves/algo_{representation}_{database}_{target_name}_{pen}.npz"
            ALGO_CURVE = np.load(ALGO_CURVE_PATH, allow_pickle=True)
            ALGO = ALGO_CURVE["mae"] * Ha2kcal
            # plot learning curve ALGO
            fig.add_trace(go.Scatter(x=N, y=ALGO, name="Frags"))

        if "sml" in curves:
            SML_CURVE_PATH = f"{parent_directory}learning_curves/sml_{representation}_{database}_{target_name}.npz"
            SML_LEARNING_CURVE = np.load(SML_CURVE_PATH, allow_pickle=True)
            SML = SML_LEARNING_CURVE["mae"] * Ha2kcal
            # plot learning curve SML
            fig.add_trace(go.Scatter(x=N, y=SML, name="SML"))

        if "fps" in curves:
            FPS_CURVE_PATH = f"{parent_directory}learning_curves/fps_{representation}_{database}_{target_name}.npz"

            FPS_LEARNING_CURVE = np.load(FPS_CURVE_PATH, allow_pickle=True)
            FPS = FPS_LEARNING_CURVE["mae"] * Ha2kcal
            # plot learning curve FPS
            fig.add_trace(go.Scatter(x=N, y=FPS, name="fps"))

        if "cur" in curves:
            CUR_CURVE_PATH = f"{parent_directory}learning_curves/cur_{representation}_{database}_{target_name}.npz"

            CUR_LEARNING_CURVE = np.load(CUR_CURVE_PATH, allow_pickle=True)
            CUR = CUR_LEARNING_CURVE["mae"] * Ha2kcal
            # plot learning curve CUR
            fig.add_trace(go.Scatter(x=N, y=CUR, name="CUR"))

        if "random" in curves:
            RANDOM_CURVE_PATH = f"{parent_directory}learning_curves/random_{representation}_{database}_{target_name}.npz"
            RANDOM_CURVE = np.load(RANDOM_CURVE_PATH, allow_pickle=True)
            MEAN_RANDOM, STD_RANDOM = (
                np.mean(RANDOM_CURVE["all_maes_random"], axis=0) * Ha2kcal,
                np.std(RANDOM_CURVE["all_maes_random"], axis=0) * Ha2kcal,
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
            yaxis=dict(tickmode="array", type="log"),
            xaxis=dict(tickmode="array", tickvals=N, type="log"),
            xaxis_title="Training set size",
            yaxis_title="MAE [kcal/mol]",
            title=f"Learning curve on target {target_name}",
        )

        SAVE_PATH = f"{parent_directory}plots/{representation}_{database}_{target_name}_{pen}.png"
        fig.write_image(SAVE_PATH)
        print(f"Saved plot to {SAVE_PATH}")

        # fig.show()

    return 0


def plots_average(
    parent_directory, database, targets, representation, pen, learning_curve_ticks, curves
):
    """
    Draw average plots of the learning curves and saves them to plots/.

    Parameters:
        parent_directory: cluster/ absolute path
        database: name of database (str) eg "qm7"
        targets: array of target names (array(str))
        representation: representation (str) eg "FCHL"
        pen: same as config["penatly"]
        in_database: whether the targets are in the database or not (bool). CUR and FPS rankings have different file names if this is the case.
    """

    N = learning_curve_ticks

    fig = go.Figure()

    if "algo" in curves:
        ALGO = []
        for target_name in targets:
            ALGO_CURVE_PATH = f"{parent_directory}learning_curves/algo_{representation}_{database}_{target_name}_{pen}.npz"
            ALGO_CURVE = np.load(ALGO_CURVE_PATH, allow_pickle=True)
            ALGO.append(ALGO_CURVE["mae"] * Ha2kcal)

        fig.add_trace(go.Scatter(x=N, y=ALGO, name="Average frags"))

    if "sml" in curves:
        SML = []
        for target_name in targets:
            SML_CURVE_PATH = f"{parent_directory}learning_curves/sml_{representation}_{database}_{target_name}.npz"
            SML_LEARNING_CURVE = np.load(SML_CURVE_PATH, allow_pickle=True)
            SML.append(SML_LEARNING_CURVE["mae"] * Ha2kcal)

        fig.add_trace(go.Scatter(x=N, y=SML, name="Average SML"))

    if "fps" in curves:
        FPS = []
        for target_name in targets:
            FPS_CURVE_PATH = f"{parent_directory}learning_curves/fps_{representation}_{database}_{target_name}.npz"
            FPS_LEARNING_CURVE = np.load(FPS_CURVE_PATH, allow_pickle=True)
            FPS.append(FPS_LEARNING_CURVE["mae"] * Ha2kcal)

        fig.add_trace(go.Scatter(x=N, y=FPS, name="Average FPS"))

    if "cur" in curves:
        CUR = []
        for target_name in targets:
            CUR_CURVE_PATH = f"{parent_directory}learning_curves/cur_{representation}_{database}_{target_name}.npz"
            CUR_LEARNING_CURVE = np.load(CUR_CURVE_PATH, allow_pickle=True)
            CUR.append(CUR_LEARNING_CURVE["mae"] * Ha2kcal)

        fig.add_trace(go.Scatter(x=N, y=CUR, name="Average CUR"))

    if "random" in curves:
        MEAN_RANDOM = []
        STD_RANDOM = []
        for target_name in targets:
            RANDOM_CURVE_PATH = f"{parent_directory}learning_curves/random_{representation}_{database}_{target_name}.npz"
            RANDOM_CURVE = np.load(RANDOM_CURVE_PATH, allow_pickle=True)
            MEAN_RANDOM.append(np.mean(RANDOM_CURVE["all_maes_random"], axis=0) * Ha2kcal)
            STD_RANDOM.append(np.std(RANDOM_CURVE["all_maes_random"], axis=0) * Ha2kcal)

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
        yaxis=dict(tickmode="array", type="log"),
        xaxis=dict(tickmode="array", tickvals=N, type="log"),
        xaxis_title="Training set size",
        yaxis_title="MAE [kcal/mol]",
        title=f"Average learning curves on {len(targets)} targets",
    )

    SAVE_PATH = f"{parent_directory}plots/{representation}_{database}_average_{pen}.png"
    fig.write_image(SAVE_PATH)
    print(f"Saved plot to {SAVE_PATH}")

    # fig.show()
    return 0

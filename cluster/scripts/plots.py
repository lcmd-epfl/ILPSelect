import numpy as np
import pandas as pd
import plotly.graph_objects as go

Ha2kcal = 627.5


###### PLOT PARAMETERS ######
# set to False to get absolute errors
PERCENTAGE_ERROR = True
# set to True to set scale to e-3 instead of e-2
MULTIPLY_BY_10 = False
# set to True iif both pen 0 and 1 are available to draw
PEN_0_AND_1 = True
# show legend switch
SHOW_LEGEND=False

def target_energy(config, target_name):
    repository_path = config["repository_folder"]
    database = config["database"]

    if config["in_database"]:
        # label of target
        Y_PATH = f"{repository_path}{database}/energies.csv"
        y_target = (
            pd.read_csv(Y_PATH).query("file == @target_name")["energy / Ha"].iloc[0]
        )

    else:
        Y_PATH = f"{repository_path}cluster/targets/energies.csv"
        y_target = (
            pd.read_csv(Y_PATH)
            .query("file == @target_name+'.xyz'")["energy / Ha"]
            .iloc[0]
        )

    # REMOVED AS PER RUBEN'S COMMENTS
    # representation = config["representation"]
    # TARGET_PATH = f"{repository_path}cluster/data/{representation}_{target_name}.npz"
    # target_info = np.load(TARGET_PATH, allow_pickle=True)
    # Q_target = target_info["ncharges"]
    # with open(f"{repository_path}cluster/data/atom_energy_coeffs.pickle", "rb") as f:
    #     atom_energy_coeffs = pickle.load(f)
    #
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
    pen = config["penalty"]
    representation = config["representation"]
    targets = config["target_names"]
    database = config["database"]
    curves = config["plots_individual"]

    # individual plots
    for target_name in targets:
        # normalization constant is energy of target
        normalization = 1
        if PERCENTAGE_ERROR:
            y_target = target_energy(config, target_name)
            normalization = (
                np.abs(y_target * Ha2kcal) / 100 / 100 / (1 + 9 * MULTIPLY_BY_10)
            )  # divide by another 10 if the graph is e-3

        print("Normalizing constant:", normalization)

        # create figure and axis
        N = learning_curve_ticks

        fig = go.Figure()

        if "algo" in curves:
            if not PEN_0_AND_1:
                ALGO_CURVE_PATH = f"{parent_directory}learning_curves/algo_{representation}_{database}_{target_name}_{pen}.npz"
                ALGO_CURVE = np.load(ALGO_CURVE_PATH, allow_pickle=True)
                ALGO = ALGO_CURVE["mae"] * Ha2kcal / normalization
                # plot learning curve ALGO
                fig.add_trace(go.Scatter(x=N, y=ALGO, name=f"ILP (p={pen})", line=dict(color="#1f77b4", width=3.5)))
                
            if PEN_0_AND_1:
                # pen 0
                ALGO_CURVE_PATH = f"{parent_directory}learning_curves/algo_{representation}_{database}_{target_name}_0.npz"
                ALGO_CURVE = np.load(ALGO_CURVE_PATH, allow_pickle=True)
                ALGO = ALGO_CURVE["mae"] * Ha2kcal / normalization
                # plot learning curve ALGO
                fig.add_trace(go.Scatter(x=N, y=ALGO, name="ILP (p=0)", line=dict(dash="dot", color="#1f77b4", width=3.5)))

                # pen 1
                ALGO_CURVE_PATH = f"{parent_directory}learning_curves/algo_{representation}_{database}_{target_name}_1.npz"
                ALGO_CURVE = np.load(ALGO_CURVE_PATH, allow_pickle=True)
                ALGO = ALGO_CURVE["mae"] * Ha2kcal / normalization
                # plot learning curve ALGO
                fig.add_trace(go.Scatter(x=N, y=ALGO, name="ILP (p=1)", line=dict(dash="solid", color="#1f77b4", width=3.5)))


        if "sml" in curves:
            SML_CURVE_PATH = f"{parent_directory}learning_curves/sml_{representation}_{database}_{target_name}.npz"
            SML_LEARNING_CURVE = np.load(SML_CURVE_PATH, allow_pickle=True)
            SML = SML_LEARNING_CURVE["mae"] * Ha2kcal / normalization
            # plot learning curve SML
            fig.add_trace(go.Scatter(x=N, y=SML, name="SML", line=dict(color="#ff7f0e", width=3.5)))

        if "fps" in curves:
            FPS_CURVE_PATH = f"{parent_directory}learning_curves/fps_{representation}_{database}_{target_name}.npz"

            FPS_LEARNING_CURVE = np.load(FPS_CURVE_PATH, allow_pickle=True)
            FPS = FPS_LEARNING_CURVE["mae"] * Ha2kcal / normalization
            # plot learning curve FPS
            fig.add_trace(go.Scatter(x=N, y=FPS, name="FPS", line=dict(color="#2ca02c", width=3.5)))

        if "cur" in curves:
            CUR_CURVE_PATH = f"{parent_directory}learning_curves/cur_{representation}_{database}_{target_name}.npz"

            CUR_LEARNING_CURVE = np.load(CUR_CURVE_PATH, allow_pickle=True)
            CUR = CUR_LEARNING_CURVE["mae"] * Ha2kcal / normalization
            # plot learning curve CUR
            fig.add_trace(go.Scatter(x=N, y=CUR, name="CUR", line=dict(color="#d62728", width=3.5)))

        if "random" in curves:
            RANDOM_CURVE_PATH = f"{parent_directory}learning_curves/random_{representation}_{database}_{target_name}.npz"
            RANDOM_CURVE = np.load(RANDOM_CURVE_PATH, allow_pickle=True)
            MEAN_RANDOM, STD_RANDOM = (
                np.mean(RANDOM_CURVE["all_maes_random"], axis=0)
                * Ha2kcal
                / normalization,
                np.std(RANDOM_CURVE["all_maes_random"], axis=0)
                * Ha2kcal
                / normalization,
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
                        width=5,
                        thickness=3,
                    ),
                    name=f"Random",
                    line=dict(color="#9467bd", width=3.5),
                )
            )

        # Update layout to make the plot more aesthetically pleasing
        fig.update_layout(
            yaxis=dict(
                type="log",
                rangemode="tozero",
                ticks="outside",
                tickformat=".1f",
                gridcolor="lightgrey",
                title_text="MAE [kcal/mol]"
                if not PERCENTAGE_ERROR
                else "MAPE [%] &times; 10<sup>-2</sup>"
                if not MULTIPLY_BY_10
                else "MAPE [%] &times; 10<sup>-3</sup>",
                title_standoff=25,
                title_font=dict(size=38, color="black", family="Arial, bold"),
            ),
            xaxis=dict(
                tickmode="array",
                tickvals=N,
                type="log",
                rangemode="tozero",
                ticks="outside",
                gridcolor="lightgrey",
                title_text="<i>N</i>",
                title_font=dict(size=38, color="black", family="Arial, bold"),
            ),
            # title=f"Average learning curves on {len(targets)} targets",
            margin=dict(t=5, r=5),  # Adjust top margin to provide space for y-axis title
            plot_bgcolor="white",
            font=dict(size=30),  # ticks font size
            height=800,  # Adjust height to make the plot less wide
            width=600,  # Adjust width to make the plot taller
            legend=dict(
                x=1,
                y=1,
                traceorder="normal",
                font=dict(family="Arial, bold", size=30, color="black"),
                bgcolor="rgba(0,0,0,0)",
            ),
            showlegend=SHOW_LEGEND,
        )

        # Update x and y axes to make lines and labels bolder
        fig.update_xaxes(showline=True, linewidth=2, linecolor="black", mirror=True)
        fig.update_yaxes(showline=True, linewidth=2, linecolor="black", mirror=True)

        if not PEN_0_AND_1:
            SAVE_PATH = f"{parent_directory}plots/{representation}_{database}_{target_name}_{pen}.svg"

        if PEN_0_AND_1:
            SAVE_PATH = f"{parent_directory}plots/{representation}_{database}_{target_name}.svg"
            
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
    database = config["database"]
    curves = config["plots_average"]

    fig = go.Figure()

    if "algo" in curves:

        if not PEN_0_AND_1:
            ALGO_PLOT(config, fig, PERCENTAGE_ERROR, MULTIPLY_BY_10, pen, "solid")

        if PEN_0_AND_1:
            ALGO_PLOT(config, fig, PERCENTAGE_ERROR, MULTIPLY_BY_10, 0, "dot")

            ALGO_PLOT(config, fig, PERCENTAGE_ERROR, MULTIPLY_BY_10, 1, "solid")

    if "sml" in curves:
        SML_PLOT(config, fig, PERCENTAGE_ERROR, MULTIPLY_BY_10)

    if "fps" in curves:
        FPS_PLOT(config, fig, PERCENTAGE_ERROR, MULTIPLY_BY_10)

    if "cur" in curves:
        CUR_PLOT(config, fig, PERCENTAGE_ERROR, MULTIPLY_BY_10)

    if "random" in curves:
        RANDOM_PLOT(config, fig, PERCENTAGE_ERROR, MULTIPLY_BY_10)

    # Update layout to make the plot more aesthetically pleasing
    fig.update_layout(
        yaxis=dict(
            type="log",
            rangemode="tozero",
            ticks="outside",
            tickformat=".1f",
            gridcolor="lightgrey",
            title_text="MAE [kcal/mol]"
            if not PERCENTAGE_ERROR
            else "MAPE [%] &times; 10<sup>-2</sup>"
            if not MULTIPLY_BY_10
            else "MAPE [%] &times; 10<sup>-3</sup>",
            title_standoff=25,
            title_font=dict(size=38, color="black", family="Arial, bold"),
        ),
        xaxis=dict(
            tickmode="array",
            tickvals=N,
            type="log",
            rangemode="tozero",
            ticks="outside",
            gridcolor="lightgrey",
            title_text="<i>N</i>",
            title_font=dict(size=38, color="black", family="Arial, bold"),
        ),
        # title=f"Average learning curves on {len(targets)} targets",
        margin=dict(t=5, r=5),  # Adjust top margin to provide space for y-axis title
        plot_bgcolor="white",
        font=dict(size=30),  # ticks font size
        height=800,  # Adjust height to make the plot less wide
        width=600,  # Adjust width to make the plot taller
        legend=dict(
            x=1,
            y=1,
            traceorder="normal",
            font=dict(family="Arial, bold", size=30, color="black"),
            bgcolor="rgba(0,0,0,0)",
        ),
        showlegend=SHOW_LEGEND,
    )

    # Update x and y axes to make lines and labels bolder
    fig.update_xaxes(showline=True, linewidth=2, linecolor="black", mirror=True)
    fig.update_yaxes(showline=True, linewidth=2, linecolor="black", mirror=True)

    config_name = config["config_name"]

    if not PEN_0_AND_1:
        SAVE_PATH = f"{parent_directory}plots/{representation}_{database}_{config_name}_average_{pen}.svg"

    if PEN_0_AND_1:
        SAVE_PATH = f"{parent_directory}plots/{representation}_{database}_{config_name}_average.svg"
    fig.write_image(SAVE_PATH)
    print(f"Saved plot to {SAVE_PATH}")

    # fig.show()
    return 0

def ALGO_PLOT(config, fig, PERCENTAGE_ERROR, MULTIPLY_BY_10, pen, style):
    """
    style: "dot" or "solid"
    """

    N = config["learning_curve_ticks"]
    parent_directory = config["current_folder"]
    representation = config["representation"]
    targets = config["plot_average_target_names"]
    database = config["database"]

    ALGO = []
    for target_name in targets:
        # normalization constant is energy of targett
        normalization = 1
        if PERCENTAGE_ERROR:
            y_target = target_energy(config, target_name)
            normalization = (
                np.abs(y_target * Ha2kcal) / 100 / 100 / (1 + 9 * MULTIPLY_BY_10)
            )  # divide by another 10 if the graph is e-3

        ALGO_CURVE_PATH = f"{parent_directory}learning_curves/algo_{representation}_{database}_{target_name}_{pen}.npz"
        ALGO_CURVE = np.load(ALGO_CURVE_PATH, allow_pickle=True)
        ALGO.append(ALGO_CURVE["mae"] * Ha2kcal / normalization)
    ALGO = np.mean(ALGO, axis=0)

    fig.add_trace(
        go.Scatter(
            x=N,
            y=ALGO,
            name=f"ILP (p={pen})",
            line=dict(dash=style, color="#1f77b4", width=3.5),
        )
    )
def SML_PLOT(config, fig, PERCENTAGE_ERROR, MULTIPLY_BY_10):

    N = config["learning_curve_ticks"]
    parent_directory = config["current_folder"]
    representation = config["representation"]
    targets = config["plot_average_target_names"]
    database = config["database"]

    SML = []
    for target_name in targets:
        # normalization constant is energy of targett
        normalization = 1
        if PERCENTAGE_ERROR:
            y_target = target_energy(config, target_name)
            normalization = (
                np.abs(y_target * Ha2kcal) / 100 / 100 / (1 + 9 * MULTIPLY_BY_10)
            )  # divide by another 10 if the graph is e-3

        SML_CURVE_PATH = f"{parent_directory}learning_curves/sml_{representation}_{database}_{target_name}.npz"
        SML_LEARNING_CURVE = np.load(SML_CURVE_PATH, allow_pickle=True)
        SML.append(SML_LEARNING_CURVE["mae"] * Ha2kcal / normalization)
    SML = np.mean(SML, axis=0)

    fig.add_trace(
        go.Scatter(x=N, y=SML, name="SML", line=dict(color="#ff7f0e", width=3.5))
    )
def FPS_PLOT(config, fig, PERCENTAGE_ERROR, MULTIPLY_BY_10):

    N = config["learning_curve_ticks"]
    parent_directory = config["current_folder"]
    representation = config["representation"]
    targets = config["plot_average_target_names"]
    database = config["database"]

    FPS = []
    for target_name in targets:
        # normalization constant is energy of target
        normalization = 1
        if PERCENTAGE_ERROR:
            y_target = target_energy(config, target_name)
            normalization = (
                np.abs(y_target * Ha2kcal) / 100 / 100 / (1 + 9 * MULTIPLY_BY_10)
            )  # divide by another 10 if the graph is e-3

        FPS_CURVE_PATH = f"{parent_directory}learning_curves/fps_{representation}_{database}_{target_name}.npz"
        FPS_LEARNING_CURVE = np.load(FPS_CURVE_PATH, allow_pickle=True)
        FPS.append(FPS_LEARNING_CURVE["mae"] * Ha2kcal / normalization)
    FPS = np.mean(FPS, axis=0)

    fig.add_trace(
        go.Scatter(x=N, y=FPS, name="FPS", line=dict(color="#2ca02c", width=3.5))
    )

def CUR_PLOT(config, fig, PERCENTAGE_ERROR, MULTIPLY_BY_10):

    N = config["learning_curve_ticks"]
    parent_directory = config["current_folder"]
    representation = config["representation"]
    targets = config["plot_average_target_names"]
    database = config["database"]
    
    CUR = []
    for target_name in targets:
        # normalization constant is energy of targett
        normalization = 1
        if PERCENTAGE_ERROR:
            y_target = target_energy(config, target_name)
            normalization = (
                np.abs(y_target * Ha2kcal) / 100 / 100 / (1 + 9 * MULTIPLY_BY_10)
            )  # divide by another 10 if the graph is e-3

        CUR_CURVE_PATH = f"{parent_directory}learning_curves/cur_{representation}_{database}_{target_name}.npz"
        CUR_LEARNING_CURVE = np.load(CUR_CURVE_PATH, allow_pickle=True)
        CUR.append(CUR_LEARNING_CURVE["mae"] * Ha2kcal / normalization)
    CUR = np.mean(CUR, axis=0)

    fig.add_trace(
        go.Scatter(x=N, y=CUR, name="CUR", line=dict(color="#d62728", width=3.5))
    )

def RANDOM_PLOT(config, fig, PERCENTAGE_ERROR, MULTIPLY_BY_10):

    N = config["learning_curve_ticks"]
    parent_directory = config["current_folder"]
    representation = config["representation"]
    targets = config["plot_average_target_names"]
    database = config["database"]
    
    MEAN_RANDOM = []
    STD_RANDOM = []
    for target_name in targets:
        # normalization constant is energy of targett
        normalization = 1
        if PERCENTAGE_ERROR:
            y_target = target_energy(config, target_name)
            normalization = (
                np.abs(y_target * Ha2kcal) / 100 / 100 / (1 + 9 * MULTIPLY_BY_10)
            )  # divide by another 10 if the graph is e-3

        RANDOM_CURVE_PATH = f"{parent_directory}learning_curves/random_{representation}_{database}_{target_name}.npz"
        RANDOM_CURVE = np.load(RANDOM_CURVE_PATH, allow_pickle=True)
        MEAN_RANDOM.append(
            np.mean(RANDOM_CURVE["all_maes_random"], axis=0)
            * Ha2kcal
            / normalization
        )
        STD_RANDOM.append(
            np.std(RANDOM_CURVE["all_maes_random"], axis=0)
            * Ha2kcal
            / normalization
        )
    MEAN_RANDOM = np.mean(MEAN_RANDOM, axis=0)
    # square std to get variance, sum because of independence, and sqrt again to get back std.
    # Var((X + Y)/2) = (Var(X) + Var(Y))/4 for X, Y independent (no covariance)
    # STD_RANDOM = np.mean(STD_RANDOM, axis=0) # wrong
    STD_RANDOM = np.sqrt(np.sum(np.array(STD_RANDOM) ** 2, axis=0)) / len(
        STD_RANDOM
    )

    fig.add_trace(
        go.Scatter(
            x=N,
            y=MEAN_RANDOM,
            error_y=dict(
                type="data",  # value of error bar given in data coordinates
                array=STD_RANDOM,
                visible=True,
                width=5,
                thickness=3,
            ),
            name="Random",
            line=dict(color="#9467bd", width=3.5),
        )
    )

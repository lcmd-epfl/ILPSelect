import pickle

import numpy as np
import pandas as pd
import plotly.graph_objects as go

Ha2kcal = 627.5


def target_energy(config, target_name):
    repository_path = config["repository_folder"]
    database = config["database"]


    if config["in_database"]:
        # label of target
        Y_PATH = f"{repository_path}{database}/energies.csv"
        y_target = pd.read_csv(Y_PATH).query("file == @target_name")["energy / Ha"].iloc[0]

    else:
        Y_PATH = f"{repository_path}cluster/targets/energies.csv"
        y_target = pd.read_csv(Y_PATH).query("file == @target_name+'.xyz'")["energy / Ha"].iloc[0]


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
            normalization = np.abs(y_target * Ha2kcal) / 100 / 100 # divide by another 100 since the graph yaxis is e-2

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
                        width=5,
                        thickness=3,
                    ),
                    name=f"Random ({CV}-fold)",
                )
            )

        # Update layout to make the plot more aesthetically pleasing
        fig.update_layout(
            yaxis=dict(
                type="log",
                rangemode='tozero',
                ticks='outside',
                tickformat=".1f",
                gridcolor='lightgrey',
                title_text="MAE [kcal/mol]" if not PERCENTAGE_ERROR else "MAPE [%] &times; 10<sup>-2</sup>",
                title_standoff=25,
                title_font=dict(size=38, color="black", family="Arial, bold"),
            ),
            xaxis=dict(
                tickmode="array",
                tickvals=N,
                type="log",
                rangemode='tozero',
                ticks='outside',
                gridcolor='lightgrey',
                title_text="<i>N</i>",
                title_font=dict(size=38, color="black", family="Arial, bold"),
            ),
            # title=f"Average learning curves on {len(targets)} targets",
            
            margin=dict(t=5, r=5),  # Adjust top margin to provide space for y-axis title
            plot_bgcolor='white',
            font=dict(size=30), # ticks font size
            height=800,  # Adjust height to make the plot less wide
            width=600,  # Adjust width to make the plot taller
            legend=dict(
                x=0.1,
                y=0.05,
                traceorder="normal",
                font=dict(
                    family="Arial, bold",
                    size=30,
                    color="black"
                ),
                bgcolor='rgba(0,0,0,0)',
            ),
        )

        # Update x and y axes to make lines and labels bolder
        fig.update_xaxes(showline=True, linewidth=2, linecolor='black', mirror=True)
        fig.update_yaxes(showline=True, linewidth=2, linecolor='black', mirror=True)

        for _, trace in enumerate(fig.data):
            # trace.line.color = f'rgba({30*i}, {30*i}, {255 - 30*i}, 1)'
            trace.line.width = 3.5

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
                normalization = np.abs(y_target * Ha2kcal) / 100 / 100 # divide by another 100 since the graph yaxis is e-2

            ALGO_CURVE_PATH = f"{parent_directory}learning_curves/algo_{representation}_{database}_{target_name}_{pen}.npz"
            ALGO_CURVE = np.load(ALGO_CURVE_PATH, allow_pickle=True)
            ALGO.append(ALGO_CURVE["mae"] * Ha2kcal / normalization)
        ALGO = np.mean(ALGO, axis=0)

        fig.add_trace(go.Scatter(x=N, y=ALGO, name="Algorithm"))

    if "sml" in curves:
        SML = []
        for target_name in targets:
            # normalization constant is energy of targett
            normalization = 1
            if PERCENTAGE_ERROR:
                y_target = target_energy(config, target_name)
                normalization = np.abs(y_target * Ha2kcal) / 100 / 100 # divide by another 100 since the graph yaxis is e-2

            SML_CURVE_PATH = f"{parent_directory}learning_curves/sml_{representation}_{database}_{target_name}.npz"
            SML_LEARNING_CURVE = np.load(SML_CURVE_PATH, allow_pickle=True)
            SML.append(SML_LEARNING_CURVE["mae"] * Ha2kcal / normalization)
        SML = np.mean(SML, axis=0)

        fig.add_trace(go.Scatter(x=N, y=SML, name="SML"))

    if "fps" in curves:
        FPS = []
        for target_name in targets:
            # normalization constant is energy of target
            normalization = 1
            if PERCENTAGE_ERROR:
                y_target = target_energy(config, target_name)
                normalization = np.abs(y_target * Ha2kcal) / 100 / 100 # divide by another 100 since the graph yaxis is e-2

            FPS_CURVE_PATH = f"{parent_directory}learning_curves/fps_{representation}_{database}_{target_name}.npz"
            FPS_LEARNING_CURVE = np.load(FPS_CURVE_PATH, allow_pickle=True)
            FPS.append(FPS_LEARNING_CURVE["mae"] * Ha2kcal / normalization)
        FPS = np.mean(FPS, axis=0)

        fig.add_trace(go.Scatter(x=N, y=FPS, name="FPS"))

    if "cur" in curves:
        CUR = []
        for target_name in targets:
            # normalization constant is energy of targett
            normalization = 1
            if PERCENTAGE_ERROR:
                y_target = target_energy(config, target_name)
                normalization = np.abs(y_target * Ha2kcal) / 100 / 100 # divide by another 100 since the graph yaxis is e-2

            CUR_CURVE_PATH = f"{parent_directory}learning_curves/cur_{representation}_{database}_{target_name}.npz"
            CUR_LEARNING_CURVE = np.load(CUR_CURVE_PATH, allow_pickle=True)
            CUR.append(CUR_LEARNING_CURVE["mae"] * Ha2kcal / normalization)
        CUR = np.mean(CUR, axis=0)

        fig.add_trace(go.Scatter(x=N, y=CUR, name="CUR"))

    if "random" in curves:
        MEAN_RANDOM = []
        STD_RANDOM = []
        for target_name in targets:
            # normalization constant is energy of targett
            normalization = 1
            if PERCENTAGE_ERROR:
                y_target = target_energy(config, target_name)
                normalization = np.abs(y_target * Ha2kcal) / 100 / 100 # divide by another 100 since the graph yaxis is e-2

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
        # STD_RANDOM = np.mean(STD_RANDOM, axis=0) # wrong
        STD_RANDOM = np.sqrt(np.sum(np.array(STD_RANDOM) ** 2, axis=0)) / len(STD_RANDOM)

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
                line=dict(color="maroon")
            )
        )

    # Update layout to make the plot more aesthetically pleasing
    fig.update_layout(
        yaxis=dict(
            type="log",
            rangemode='tozero',
            ticks='outside',
            tickformat=".1f",
            gridcolor='lightgrey',
            title_text="MAE [kcal/mol]" if not PERCENTAGE_ERROR else "MAPE [%] &times; 10<sup>-2</sup>",
            title_standoff=25,
            title_font=dict(size=38, color="black", family="Arial, bold"),
        ),
        xaxis=dict(
            tickmode="array",
            tickvals=N,
            type="log",
            rangemode='tozero',
            ticks='outside',
            gridcolor='lightgrey',
            title_text="<i>N</i>",
            title_font=dict(size=38, color="black", family="Arial, bold"),
        ),
        # title=f"Average learning curves on {len(targets)} targets",
        
        margin=dict(t=5, r=5),  # Adjust top margin to provide space for y-axis title
        plot_bgcolor='white',
        font=dict(size=30), # ticks font size
        height=800,  # Adjust height to make the plot less wide
        width=600,  # Adjust width to make the plot taller
        legend=dict(
            x=1.1,
            y=1.1,
            traceorder="normal",
            font=dict(
                family="Arial, bold",
                size=30,
                color="black"
            ),
            bgcolor='rgba(0,0,0,0)',
        ),
        # showlegend=False
    )

    # Update x and y axes to make lines and labels bolder
    fig.update_xaxes(showline=True, linewidth=2, linecolor='black', mirror=True)
    fig.update_yaxes(showline=True, linewidth=2, linecolor='black', mirror=True)

    colours = [
        '#1f77b4',  # muted blue
        '#ff7f0e',  # safety orange
        '#2ca02c',  # cooked asparagus green
        '#d62728',  # brick red
        '#9467bd',  # muted purple
        '#8c564b',  # chestnut brown
        '#e377c2',  # raspberry yogurt pink
        '#7f7f7f',  # middle gray
        '#bcbd22',  # curry yellow-green
        '#17becf'   # blue-teal
    ]

    for i, trace in enumerate(fig.data):
        trace.line.color = colours[i]
        trace.line.width = 3.5


    SAVE_PATH = f"{parent_directory}plots/{representation}_{database}_average_{pen}.png"
    fig.write_image(SAVE_PATH)
    print(f"Saved plot to {SAVE_PATH}")

    # fig.show()
    return 0

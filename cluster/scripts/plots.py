import matplotlib.pyplot as plt
import numpy as np

# matplotlib font size
plt.rcParams.update({"font.size": 18})

Ha2kcal = 627.5


def plots(parent_directory, database, targets, representation, config, algorithms):
    """
    Draw combined plots and average plot of the learning curves and saves them to plots/.

    Parameters:
        parent_directory: cluster/ absolute path
        database: name of database (str) eg "qm7"
        targets: array of target names (array(str))
        representation: representation (str) eg "FCHL"
        config: config directory. Must include key "penalty"
        algorithms: list of algorithms to test (array(str)). WARNING only ["fragments", "sml"] is handled.
    """

    for algorithm in algorithms:
        assert algorithm in ["fragments", "sml"], "only fragments and sml algorithms are handled"

    pen = config["penalty"]

    # individual plots
    for target_name in targets:
        ALGO_CURVE_PATH = f"{parent_directory}learning_curves/algo_{representation}_{database}_{target_name}_{pen}.npz"
        SML_CURVE_PATH = (
            f"{parent_directory}learning_curves/sml_{representation}_{database}_{target_name}.npz"
        )

        SML_LEARNING_CURVE = np.load(SML_CURVE_PATH)
        ALGO_CURVE = np.load(ALGO_CURVE_PATH)
        MEAN_RANDOM, STD_RANDOM = (
            np.mean(SML_LEARNING_CURVE["all_maes_random"], axis=0) * Ha2kcal,
            np.std(SML_LEARNING_CURVE["all_maes_random"], axis=0) * Ha2kcal,
        )
        SML = SML_LEARNING_CURVE["mae"] * Ha2kcal
        FRAG = ALGO_CURVE["mae"] * Ha2kcal
        N = SML_LEARNING_CURVE["train_sizes"]
        # create figure and axis
        fig, ax = plt.subplots(figsize=(11, 6))
        # plot learning curve random with std as error bars
        ax.errorbar(N, MEAN_RANDOM, yerr=STD_RANDOM, fmt="o-", label="Random")
        # plot learning curve SML
        ax.plot(N, SML, "o-", label="SML")
        # plot learning curve FRAG
        ax.plot(N, FRAG, "o-", label=f"Fragments algo pen {pen}")
        # set axis labels
        ax.set_xlabel("Training set size")
        ax.set_ylabel("MAE [kcal/mol]")
        # set log scale on x axis
        ax.set_xscale("log")
        # set log scale on y axis
        ax.set_yscale("log")
        # legend
        ax.legend()
        # title
        plt.title(f"Learning curve on target {target_name}")
        # turn minor ticks off
        ax.minorticks_off()
        # make x ticks as N
        ax.set_xticks(N)
        ax.set_xticklabels(N)
        # grid on
        ax.grid()
        # save figure
        SAVE_PATH = f"{parent_directory}plots/{representation}_{database}_{target_name}_{pen}.png"
        fig.savefig(SAVE_PATH, dpi=300)
        print(f"Saved plot to {SAVE_PATH}")

    # average plot
    MEAN_RANDOM = []
    STD_RANDOM = []

    SML = []
    FRAG = []
    N = np.array([2**i for i in range(4, 11)])

    for target_name in targets:
        ALGO_CURVE_PATH = f"{parent_directory}learning_curves/algo_{representation}_{database}_{target_name}_{pen}.npz"
        SML_CURVE_PATH = (
            f"{parent_directory}learning_curves/sml_{representation}_{database}_{target_name}.npz"
        )

        SML_LEARNING_CURVE = np.load(SML_CURVE_PATH)
        ALGO_CURVE = np.load(ALGO_CURVE_PATH)

        MEAN_RANDOM.append(np.mean(SML_LEARNING_CURVE["all_maes_random"], axis=0) * Ha2kcal)
        STD_RANDOM.append(np.std(SML_LEARNING_CURVE["all_maes_random"], axis=0) * Ha2kcal)
        SML.append(SML_LEARNING_CURVE["mae"] * Ha2kcal)
        FRAG.append(ALGO_CURVE["mae"] * Ha2kcal)

    # TODO: not sure average of STDs makes sense
    MEAN_RANDOM = np.mean(MEAN_RANDOM, axis=0)
    STD_RANDOM = np.mean(STD_RANDOM, axis=0)

    SML = np.mean(SML, axis=0)
    FRAG = np.mean(FRAG, axis=0)

    # create figure and axis
    fig, ax = plt.subplots(figsize=(11, 6))
    # plot learning curve random with std as error bars
    ax.errorbar(N, MEAN_RANDOM, yerr=STD_RANDOM, fmt="o-", label="Average random")
    # plot learning curve SML
    ax.plot(N, SML, "o-", label="Average SML")
    # plot learning curve FRAG
    ax.plot(N, FRAG, "o-", label=f"Average frags")
    # set axis labels
    ax.set_xlabel("Training set size")
    ax.set_ylabel("MAE [kcal/mol]")
    # set log scale on x axis
    ax.set_xscale("log")
    # set log scale on y axis
    ax.set_yscale("log")
    # legend
    ax.legend()
    # title
    plt.title(f"Average learning curves")
    # turn minor ticks off
    ax.minorticks_off()
    # make x ticks as N
    ax.set_xticks(N)
    ax.set_xticklabels(N)
    # grid on
    ax.grid()
    # save figure
    SAVE_PATH = f"{parent_directory}plots/{representation}_{database}_average_{pen}.png"
    fig.savefig(SAVE_PATH, dpi=300)
    print(f"Saved plot to {SAVE_PATH}")

    return 0

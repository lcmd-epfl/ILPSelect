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
        RANDOM_CURVE_PATH = f"{parent_directory}learning_curves/random_{representation}_{database}_{target_name}.npz"

        SML_LEARNING_CURVE = np.load(SML_CURVE_PATH, allow_pickle=True)
        ALGO_CURVE = np.load(ALGO_CURVE_PATH, allow_pickle=True)
        RANDOM_CURVE = np.load(RANDOM_CURVE_PATH, allow_pickle=True)
        MEAN_RANDOM, STD_RANDOM = (
            np.mean(RANDOM_CURVE["all_maes_random"], axis=0) * Ha2kcal,
            np.std(RANDOM_CURVE["all_maes_random"], axis=0) * Ha2kcal,
        )
        SML = SML_LEARNING_CURVE["mae"] * Ha2kcal
        ALGO = ALGO_CURVE["mae"] * Ha2kcal
        N = SML_LEARNING_CURVE["train_sizes"]
        # create figure and axis
        fig, ax = plt.subplots(figsize=(11, 6))
        # plot learning curve random with std as error bars
        CV = len(RANDOM_CURVE["all_maes_random"])
        ax.errorbar(N, MEAN_RANDOM, yerr=STD_RANDOM, fmt="o-", label=f"Random ({CV}-fold)")
        # plot learning curve SML
        ax.plot(N, SML, "o-", label="SML")
        # plot learning curve ALGO
        ax.plot(N, ALGO, "o-", label=f"Fragments algo pen {pen}")
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
    ALGO = []
    N = np.array([2**i for i in range(4, 11)])

    for target_name in targets:
        ALGO_CURVE_PATH = f"{parent_directory}learning_curves/algo_{representation}_{database}_{target_name}_{pen}.npz"
        SML_CURVE_PATH = (
            f"{parent_directory}learning_curves/sml_{representation}_{database}_{target_name}.npz"
        )
        RANDOM_CURVE_PATH = f"{parent_directory}learning_curves/random_{representation}_{database}_{target_name}.npz"

        SML_LEARNING_CURVE = np.load(SML_CURVE_PATH, allow_pickle=True)
        ALGO_CURVE = np.load(ALGO_CURVE_PATH, allow_pickle=True)
        RANDOM_CURVE = np.load(RANDOM_CURVE_PATH, allow_pickle=True)

        MEAN_RANDOM.append(np.mean(RANDOM_CURVE["all_maes_random"], axis=0) * Ha2kcal)
        STD_RANDOM.append(np.std(RANDOM_CURVE["all_maes_random"], axis=0) * Ha2kcal)
        SML.append(SML_LEARNING_CURVE["mae"] * Ha2kcal)
        ALGO.append(ALGO_CURVE["mae"] * Ha2kcal)

    # TODO: not sure average of STDs makes sense
    MEAN_RANDOM = np.mean(MEAN_RANDOM, axis=0)
    STD_RANDOM = np.mean(STD_RANDOM, axis=0)

    SML = np.mean(SML, axis=0)
    ALGO = np.mean(ALGO, axis=0)

    # create figure and axis
    fig, ax = plt.subplots(figsize=(11, 6))
    # plot learning curve random with std as error bars
    CV = len(RANDOM_CURVE["all_maes_random"])
    ax.errorbar(N, MEAN_RANDOM, yerr=STD_RANDOM, fmt="o-", label=f"Average random ({CV}-fold)")
    # plot learning curve SML
    ax.plot(N, SML, "o-", label="Average SML")
    # plot learning curve ALGO
    ax.plot(N, ALGO, "o-", label=f"Average frags")
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
    plt.title(f"Average learning curves on {len(targets)} targets")
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
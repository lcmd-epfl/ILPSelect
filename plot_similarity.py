import os
import numpy as np
from sklearn.metrics import pairwise_distances
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from matplotlib.colors import Normalize
from openTSNE import TSNE

np.random.seed(20)
plt.rcParams["figure.figsize"] = (7, 4.8)
import matplotlib
import pickle

matplotlib.rcParams.update({"font.size": 20})

from scripts.kernels import get_local_kernel
import seaborn as sns
import argparse as ap
import pandas as pd

pt = {"C": 6, "N": 7, "O": 8, "S": 16, "F": 9, "H": 1}


def plot_tsne(
    x,
    y,
    target_name,
    training_name,
    ax=None,
    title=None,
    draw_legend=True,
    draw_centers=False,
    draw_cluster_labels=False,
    colors=None,
    legend_kwargs=None,
    label_order=None,
    **kwargs,
):
    import matplotlib

    if ax is None:
        _, ax = matplotlib.pyplot.subplots(figsize=(8, 8))

    if title is not None:
        ax.set_title(title)

    plot_params = {"alpha": kwargs.get("alpha", 0.6), "s": kwargs.get("s", 1)}

    # Set up sizes and alphas via trick
    s = np.array((y + 1) * (y + 18), dtype=int)
    alphas = np.clip((y + 1) * 0.4, a_min=0, a_max=1)

    # Create main plot
    if label_order is not None:
        assert all(np.isin(np.unique(y), label_order))
        classes = [l for l in label_order if l in np.unique(y)]
    else:
        classes = np.unique(y)
    if colors is None:
        default_colors = matplotlib.rcParams["axes.prop_cycle"]
        colors = {k: v["color"] for k, v in zip(classes, default_colors())}

    point_colors = list(map(colors.get, y))

    ax.scatter(
        x[:, 0],
        x[:, 1],
        c=point_colors,
        s=s,
        alpha=alphas,
        edgecolors=point_colors,
        rasterized=True,
    )  # , **plot_params)

    # Plot mediods
    if draw_centers:
        centers = []
        for yi in classes:
            mask = yi == y
            centers.append(np.median(x[mask, :2], axis=0))
        centers = np.array(centers)

        center_colors = list(map(colors.get, classes))
        ax.scatter(
            centers[:, 0], centers[:, 1], c=center_colors, s=48, alpha=1, edgecolor="k"
        )

        # Draw mediod labels
        if draw_cluster_labels:
            for idx, label in enumerate(classes):
                ax.text(
                    centers[idx, 0],
                    centers[idx, 1] + 2.2,
                    label,
                    fontsize=kwargs.get("fontsize", 6),
                    horizontalalignment="center",
                )

    # Hide ticks and axis
    ax.set_xticks([]), ax.set_yticks([]), ax.axis("off")

    if draw_legend:
        legend_handles = [
            matplotlib.lines.Line2D(
                [],
                [],
                marker="s",
                color="w",
                markerfacecolor=colors[yi],
                ms=10,
                alpha=1,
                linewidth=0,
                label=yi,
                markeredgecolor="k",
            )
            for yi in classes
        ]
        legend_kwargs_ = dict(loc="center left", bbox_to_anchor=(1, 0.5), frameon=False)
        if legend_kwargs is not None:
            legend_kwargs_.update(legend_kwargs)
        ax.legend(handles=legend_handles, **legend_kwargs_)

    # plt.savefig(f"tsne.pdf")
    plt.savefig(f"interpret_figs/tsne_{target_name}_{training_name}.pdf")
    # plt.show()
    return


def filter(ar):
    return ar[np.isfinite(ar)]


def parse_args():
    parser = ap.ArgumentParser()
    parser.add_argument("-t", "--target", default="sildenafil")  # can be all
    parser.add_argument("-m", "--min", action="store_true")  # instead of all distances
    parser.add_argument("-d", "--database", default="drugs")
    parser.add_argument("-s", "--size", action="store_true")  # size plot instead
    args = parser.parse_args()
    return args


def compute_pairwise_distances(
    mixed_Xs, X_target, atomtypes_database, atomtypes_target, option="all"
):
    """
    Compute the squared Euclidean distances between each target atom and all atoms in the database.

    Parameters:
    mixed_Xs (list of list of lists): The database of atomic environments with mixed dimensions.
    X_target (numpy.ndarray): The feature vectors for the target atoms with dimensions (target_dim, feature_size).
    atomtypes_database (list of lists): Atom types for each atom in the database with mixed dimensions.
    atomtypes_target (list): Atom types for each atom in the target molecule.
    option (str): 'all' to return all distances, 'min' to return only the minimum distances, 'max' to return maximum distances.

    Returns:
    numpy.ndarray: A 2D array of pairwise distances with dimensions (target_dim, total_database_atoms) if option is 'all'.
                   A 2D array of minimum distances with dimensions (target_dim, n_samples) if option is 'min'.
    """
    target_dim, target_feature_size = X_target.shape
    n_samples = len(mixed_Xs)

    # Flatten the database to get the total number of atoms and their atom types
    flattened_Xs = [
        np.array(atom_features) for sublist in mixed_Xs for atom_features in sublist
    ]
    flattened_atomtypes = [
        atomtype for sublist in atomtypes_database for atomtype in sublist
    ]
    total_database_atoms = len(flattened_Xs)

    if option == "all":
        # Initialize the distance matrix for 'all' option
        distances = np.full((target_dim, total_database_atoms), np.inf)

        # Compute distances
        for i in range(target_dim):
            for j, (db_atom_features, db_atomtype) in enumerate(
                zip(flattened_Xs, flattened_atomtypes)
            ):
                if atomtypes_target[i] == db_atomtype:
                    distances[i, j] = (
                        np.linalg.norm(X_target[i] - db_atom_features) ** 2
                    )

        return distances

    if option == "min" or option == "delta":
        # Initialize the distance matrix for 'min' option
        min_distances = np.zeros((target_dim, n_samples))

        # Compute minimum distances
        for i in range(target_dim):
            for sample_index, (sublist, sublist_atomtypes) in enumerate(
                zip(mixed_Xs, atomtypes_database)
            ):
                min_distance = np.inf
                for atom_features, atomtype in zip(sublist, sublist_atomtypes):
                    if atomtypes_target[i] == atomtype:
                        db_atom_features = np.array(atom_features)
                        distance = np.linalg.norm(X_target[i] - db_atom_features)
                        if distance < min_distance:
                            min_distance = distance
                if min_distance == np.inf:
                    # print("No matching atom at all! Weird")
                    min_distance = (
                        np.nan
                    )  # Handle cases where no matching atom type is found
                min_distances[i, sample_index] = min_distance
        if option == "min":
            return min_distances

    if option == "max" or option == "delta":
        # Initialize the distance matrix for 'min' option
        max_distances = np.zeros((target_dim, n_samples))

        # Compute maximum distances
        for i in range(target_dim):
            for sample_index, (sublist, sublist_atomtypes) in enumerate(
                zip(mixed_Xs, atomtypes_database)
            ):
                max_distance = -np.inf
                for atom_features, atomtype in zip(sublist, sublist_atomtypes):
                    if atomtypes_target[i] == atomtype:
                        db_atom_features = np.array(atom_features)
                        distance = np.linalg.norm(X_target[i] - db_atom_features)
                        if distance > max_distance:
                            max_distance = distance
                if max_distance == -np.inf:
                    # print("No matching atom at all! Weird")
                    max_distance = (
                        np.nan
                    )  # Handle cases where no matching atom type is found
                max_distances[i, sample_index] = max_distance

        if option == "max":
            return max_distances

    if option == "delta":
        return max_distances - min_distances


def compute_pairwise_distances_database(mixed_Xs, atomtypes_database, option="all"):
    n_samples = len(mixed_Xs)

    # Flatten the database to get the total number of atoms and their atom types
    flattened_Xs = [
        np.array(atom_features) for sublist in mixed_Xs for atom_features in sublist
    ]
    flattened_atomtypes = [
        atomtype for sublist in atomtypes_database for atomtype in sublist
    ]
    total_database_atoms = len(flattened_Xs)

    # Initialize the distance matrix for 'all' option
    distances = np.zeros((total_database_atoms, total_database_atoms))

    # Compute distances
    for i in range(total_database_atoms):
        for j, (db_atom_features, db_atomtype) in enumerate(
            zip(flattened_Xs, flattened_atomtypes)
        ):
            if i == j:
                pass
            elif j > i:
                pass
            elif flattened_atomtypes[i] == db_atomtype:
                distances[j, i] = distances[i, j] = np.linalg.norm(
                    flattened_Xs[i] - db_atom_features
                )
    distances = distances[np.triu_indices(total_database_atoms, k=1)]
    # print(distances.shape, distances.size)
    return distances


def local_global_sim(Xs, X_target, Qs, Q_target, sigma=1, normalise=True):
    K = get_local_kernel(
        Xs, np.array([X_target]), Qs, np.array([Q_target]), sigma=sigma
    ).flatten()
    # flatten is assuming there was a single target X_target
    nats_db = np.array([len(x) for x in Qs])
    assert len(K) == len(nats_db)
    if normalise:
        return K / (nats_db * len(Q_target))
    else:
        return K


def get_heavy_rep(ncharges, reps):
    h_reps = []
    h_ncharges = []
    for i, mol_ncharges in enumerate(ncharges):
        h_filter = np.where(mol_ncharges != 1)
        X = reps[i][h_filter]
        h_ncharges.append(mol_ncharges[h_filter])
        h_reps.append(X)

    return h_reps, h_ncharges


def load_reps_target(target):
    # LOAD REPS/NCHARGES
    target_data = np.load(f"data/FCHL_{target}.npz", allow_pickle=True)
    target_rep = target_data["rep"]
    target_ncharges = target_data["ncharges"]
    h_indices = np.where(target_ncharges != 1)[0]
    h_target_rep = target_rep[h_indices]
    h_target_ncharges = target_ncharges[h_indices]
    return target_rep, target_ncharges, h_target_rep, h_target_ncharges


def load_qm7(target, database="drugs"):
    qm7_data = np.load(f"data/FCHL_qm7_qm7{database}.npz", allow_pickle=True)
    qm7_reps = qm7_data["reps"]
    qm7_ncharges = qm7_data["ncharges"]
    qm7_labels = qm7_data["labels"]
    check_1 = None
    check_2 = None
    check_3 = None

    # now get selection based on target molecule / method
    algo_1_indices = np.load(
        f"rankings/algo_FCHL_qm7_{target}_1.npy", allow_pickle=True
    )
    algo_1_ncharges, algo_1_reps, y_algo_1 = (
        qm7_ncharges[algo_1_indices],
        qm7_reps[algo_1_indices],
        qm7_labels[algo_1_indices],
    )
    h_algo_1_reps, h_algo_1_ncharges = get_heavy_rep(algo_1_ncharges, algo_1_reps)
    sizes_algo_1 = get_molecule_sizes(algo_1_ncharges)

    algo_0_indices = np.load(
        f"rankings/algo_FCHL_qm7_{target}_0.npy", allow_pickle=True
    )
    algo_0_ncharges, algo_0_reps, y_algo_0 = (
        qm7_ncharges[algo_0_indices],
        qm7_reps[algo_0_indices],
        qm7_labels[algo_0_indices],
    )
    h_algo_0_reps, h_algo_0_ncharges = get_heavy_rep(algo_0_ncharges, algo_0_reps)
    sizes_algo_0 = get_molecule_sizes(algo_0_ncharges)

    cur_indices = np.load(f"rankings/cur_FCHL_qm7.npy", allow_pickle=True)
    if check_1 is None:
        check_1 = np.sort(cur_indices)
    else:
        test = np.sort(cur_indices)
        assert (check_1 == test).all()
        check_1 = test
    cur_ncharges, cur_reps, y_cur = (
        qm7_ncharges[cur_indices],
        qm7_reps[cur_indices],
        qm7_labels[cur_indices],
    )
    h_cur_reps, h_cur_ncharges = get_heavy_rep(cur_ncharges, cur_reps)
    sizes_cur = get_molecule_sizes(cur_ncharges)

    fps_indices = np.load(f"rankings/fps_FCHL_qm7.npz", allow_pickle=True)["arr_6"]
    if check_2 is None:
        check_2 = fps_indices
    else:
        test = np.sort(fps_indices)
        assert (check_2 == test).all()
        check_2 = test
    fps_ncharges, fps_reps, y_fps = (
        qm7_ncharges[fps_indices],
        qm7_reps[fps_indices],
        qm7_labels[fps_indices],
    )
    h_fps_reps, h_fps_ncharges = get_heavy_rep(fps_ncharges, fps_reps)
    sizes_fps = get_molecule_sizes(fps_ncharges)

    sml_indices = np.load(f"rankings/sml_FCHL_qm7_{target}.npy", allow_pickle=True)
    if check_3 is None:
        check_2 = sml_indices
    else:
        test = np.sort(sml_indices)
        assert (check_3 == test).all()
        check_3 = test
    sml_ncharges, sml_reps, y_sml = (
        qm7_ncharges[sml_indices],
        qm7_reps[sml_indices],
        qm7_labels[sml_indices],
    )
    h_sml_reps, h_sml_ncharges = get_heavy_rep(sml_ncharges, sml_reps)
    sizes_sml = get_molecule_sizes(sml_ncharges)

    random_indices = np.random.choice(
        np.arange(len(qm7_labels)), size=1024, replace=False
    )
    random_ncharges, random_reps, y_random = (
        qm7_ncharges[random_indices],
        qm7_reps[random_indices],
        qm7_labels[random_indices],
    )
    h_random_reps, h_random_ncharges = get_heavy_rep(random_ncharges, random_reps)
    sizes_random = get_molecule_sizes(random_ncharges)

    out = (
        algo_1_ncharges,
        algo_1_reps,
        sizes_algo_1,
        h_algo_1_ncharges,
        h_algo_1_reps,
        algo_1_indices,
        algo_0_ncharges,
        algo_0_reps,
        sizes_algo_0,
        h_algo_0_ncharges,
        h_algo_0_reps,
        algo_0_indices,
        cur_ncharges,
        cur_reps,
        sizes_cur,
        h_cur_ncharges,
        h_cur_reps,
        cur_indices,
        fps_ncharges,
        fps_reps,
        sizes_fps,
        h_fps_ncharges,
        h_fps_reps,
        fps_indices,
        sml_ncharges,
        sml_reps,
        sizes_sml,
        h_sml_ncharges,
        h_sml_reps,
        sml_indices,
        random_ncharges,
        random_reps,
        sizes_random,
        h_random_ncharges,
        h_random_reps,
        random_indices,
    )
    return out, qm7_ncharges, qm7_reps


def get_molecule_sizes(ncharges_list, heavy=True):
    sizes = []
    for ncharges in ncharges_list:
        if heavy:
            ncharges = [x for x in ncharges if x != 1]
        size = len(ncharges)
        sizes.append(size)
    return sizes


def size_plot(
    sizes_algo_0, sizes_algo_1, sizes_random, sizes_cur, sizes_sml, sizes_fps
):
    fig, ax = plt.subplots(nrows=1, ncols=1)
    ax.hist(
        sizes_algo_0,
        label="ILP(p=0)",
        color=colors[0],
        edgecolor=colors[0],
        alpha=0.5,
        hatch="**",
        bins=100,
    )
    ax.hist(
        sizes_algo_1,
        label="ILP(p=1)",
        color=colors[1],
        edgecolor=colors[1],
        alpha=0.5,
        hatch="//",
        bins=100,
    )
    ax.hist(
        sizes_random,
        label="random",
        color=colors[2],
        edgecolor=colors[2],
        alpha=0.5,
        bins=100,
    )
    ax.hist(
        sizes_cur,
        label="CUR",
        color=colors[3],
        edgecolor=colors[3],
        alpha=0.5,
        bins=100,
    )
    ax.hist(
        sizes_sml,
        label="SML",
        color=colors[4],
        edgecolor=colors[4],
        alpha=0.5,
        bins=100,
    )
    ax.hist(
        sizes_fps,
        label="FPS",
        color=colors[5],
        edgecolor=colors[5],
        alpha=0.5,
        bins=100,
    )
    ax.set_xlabel("Number of heavy atoms")
    ax.set_ylabel("Count")
    plt.legend()
    plt.savefig(f"interpret_figs/{target}_sizes.pdf")
    plt.show()
    return


def combined_size_plot_stacked(targets_data, bin_width=0.1, database="drugs"):
    """
    Plot the sizes (number of heavy atoms) for a set of target molecules and their associated subsets of molecules.
    The bars for different methods are stacked vertically in a single plot.

    Parameters:
    targets_data (list of dict): List of dictionaries where each dictionary contains the following keys:
                                 - 'sizes_algo_0': List of sizes for the ILP(p=0) subset.
                                 - 'sizes_algo_1': List of sizes for the ILP(p=1) subset.
                                 - 'sizes_random': List of sizes for the random subset.
                                 - 'sizes_cur': List of sizes for the CUR subset.
                                 - 'sizes_sml': List of sizes for the SML subset.
                                 - 'sizes_fps': List of sizes for the FPS subset.
                                 - 'target_name': A name or identifier for the target molecule.
    bin_width (int): Width of the bins for the histogram.

    Returns:
    None
    """

    # Initialize lists to store the sizes
    all_sizes_algo_0 = []
    all_sizes_algo_1 = []
    all_sizes_random = []
    all_sizes_cur = []
    all_sizes_sml = []
    all_sizes_fps = []

    for target_data in targets_data:
        all_sizes_algo_0.extend(target_data["sizes_algo_0"])
        all_sizes_algo_1.extend(target_data["sizes_algo_1"])
        all_sizes_random.extend(target_data["sizes_random"])
        all_sizes_cur.extend(target_data["sizes_cur"])
        all_sizes_sml.extend(target_data["sizes_sml"])
        all_sizes_fps.extend(target_data["sizes_fps"])

    # Create a figure
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 8))

    # Plot each histogram with a slight offset
    ax.hist(
        np.array(all_sizes_algo_0) - bin_width * 2.5,
        bins=50,
        label="ILP(p=0)",
        color=colors[0],
        alpha=0.8,
        orientation="horizontal",
        hatch="***",
        edgecolor="black",
    )
    ax.hist(
        np.array(all_sizes_algo_1) - bin_width * 1.5,
        bins=50,
        label="ILP(p=1)",
        color=colors[1],
        alpha=0.8,
        orientation="horizontal",
        hatch="///",
        edgecolor="black",
    )
    ax.hist(
        np.array(all_sizes_random) - bin_width * 0.5,
        bins=50,
        label="Random",
        color=colors[2],
        edgecolor=colors[2],
        alpha=0.8,
        orientation="horizontal",
    )
    ax.hist(
        np.array(all_sizes_cur) + bin_width * 0.5,
        bins=50,
        label="CUR",
        color=colors[3],
        edgecolor=colors[3],
        alpha=0.8,
        orientation="horizontal",
    )
    ax.hist(
        np.array(all_sizes_sml) + bin_width * 1.5,
        bins=50,
        label="SML",
        color=colors[4],
        edgecolor=colors[4],
        alpha=0.8,
        orientation="horizontal",
    )
    ax.hist(
        np.array(all_sizes_fps) + bin_width * 2.5,
        bins=50,
        label="FPS",
        color=colors[5],
        edgecolor=colors[5],
        alpha=0.8,
        orientation="horizontal",
    )

    # Labeling the plot
    ax.set_ylabel("Number of heavy atoms")
    ax.set_xlabel("Count")
    ax.set_yticks([1, 2, 3, 4, 5, 6, 7])
    ax.set_yticklabels(["1", "2", "3", "4", "5", "6", "7"])
    # ax.set_yticks(bins + bin_width / 2)  # Center the tick labels
    ax.set_xscale("log")
    plt.legend(loc="lower right")
    plt.savefig(
        f"interpret_figs/combined_size_plot_stacked_{database}.pdf", format="pdf"
    )
    plt.show()
    return


def distance_plot(
    h_algo_0_reps,
    h_algo_0_ncharges,
    h_algo_1_reps,
    h_algo_1_ncharges,
    h_random_reps,
    h_random_ncharges,
    h_cur_reps,
    h_cur_ncharges,
    h_sml_reps,
    h_sml_ncharges,
    h_fps_reps,
    h_fps_ncharges,
    h_target_rep,
    h_target_ncharges,
    option="all",
):
    algo_0_d = np.concatenate(
        compute_pairwise_distances(
            h_algo_0_reps,
            h_target_rep,
            h_algo_0_ncharges,
            h_target_ncharges,
            option=option,
        ),
        axis=0,
    )
    algo_1_d = np.concatenate(
        compute_pairwise_distances(
            h_algo_1_reps,
            h_target_rep,
            h_algo_1_ncharges,
            h_target_ncharges,
            option=option,
        ),
        axis=0,
    )
    random_d = np.concatenate(
        compute_pairwise_distances(
            h_random_reps,
            h_target_rep,
            h_random_ncharges,
            h_target_ncharges,
            option=option,
        ),
        axis=0,
    )
    cur_d = np.concatenate(
        compute_pairwise_distances(
            h_cur_reps, h_target_rep, h_cur_ncharges, h_target_ncharges, option=option
        ),
        axis=0,
    )
    sml_d = np.concatenate(
        compute_pairwise_distances(
            h_sml_reps, h_target_rep, h_sml_ncharges, h_target_ncharges, option=option
        ),
        axis=0,
    )
    fps_d = np.concatenate(
        compute_pairwise_distances(
            h_fps_reps, h_target_rep, h_fps_ncharges, h_target_ncharges, option=option
        ),
        axis=0,
    )

    fig, ax = plt.subplots(nrows=1, ncols=1)
    sns.kdeplot(algo_0_d, label="ILP(p=0)", color=colors[0], linestyle="dashed")
    sns.kdeplot(algo_1_d, label="ILP(p=1)", color=colors[1])
    sns.kdeplot(random_d, label="random", color=colors[2])
    sns.kdeplot(cur_d, label="CUR", color=colors[3])
    sns.kdeplot(sml_d, label="SML", color=colors[4])
    sns.kdeplot(fps_d, label="FPS", color=colors[5])
    ax.set_xlim(0, 13)
    ax.set_xlabel("Euclidean distance to target atoms")
    plt.legend()
    plt.savefig(f"interpret_figs/{target}_dist_targets_{option}.pdf")
    plt.show()
    return


def tsne_plots(
    qm7_reps,
    qm7_ncharges,
    targets_data,
    database="drugs",
    option="all",
    global_rep=True,
):
    """
    Plot all distances observed for a set of target molecules and their associated subsets of molecules in a single plot.

    Parameters:
    targets_data (list of dict): List of dictionaries where each dictionary contains the following keys:
                                 - 'target_rep': The feature vectors for the target atoms.
                                 - 'target_ncharges': The atom types for the target atoms.
                                 - 'h_algo_0_reps': The feature vectors for the ILP(p=0) subset.
                                 - 'h_algo_0_ncharges': The atom types for the ILP(p=0) subset.
                                 - 'h_algo_1_reps': The feature vectors for the ILP(p=1) subset.
                                 - 'h_algo_1_ncharges': The atom types for the ILP(p=1) subset.
                                 - 'h_random_reps': The feature vectors for the random subset.
                                 - 'h_random_ncharges': The atom types for the random subset.
                                 - 'h_cur_reps': The feature vectors for the CUR subset.
                                 - 'h_cur_ncharges': The atom types for the CUR subset.
                                 - 'h_sml_reps': The feature vectors for the SML subset.
                                 - 'h_sml_ncharges': The atom types for the SML subset.
                                 - 'h_fps_reps': The feature vectors for the FPS subset.
                                 - 'h_fps_ncharges': The atom types for the FPS subset.
                                 - 'target_name': A name or identifier for the target molecule.
    option (str): Option for computing distances ('all' or 'min').

    Returns:
    None
    """
    if global_rep:
        # Embedding on qm7
        # print(qm7_reps.shape, qm7_reps.size)
        qm7_reps = qm7_reps.sum(axis=1)
        # print(qm7_reps.shape, qm7_reps.size)

        tsne = TSNE(
            perplexity=100,
            metric="euclidean",
            n_jobs=1,
            random_state=42,
            verbose=True,
            early_exaggeration_iter=50,
        )
        # early_exaggeration_iter=10,
        # n_iter=10,
        e_train = tsne.fit(qm7_reps)
        x_qm7 = e_train.transform(qm7_reps)
        # plot_tsne(x_qm7, np.zeros((x_qm7.shape[0])))
        training_set_names = [
            "h_algo_0_reps",
            "h_algo_1_reps",
            "h_random_reps",
            "h_cur_reps",
            "h_sml_reps",
            "h_fps_reps",
        ]

        for t, target_data in enumerate(targets_data):
            for training_name in training_set_names:
                target_rep = target_data["target_rep"]
                target_ncharges = target_data["target_ncharges"]
                target_name = target_data["target_name"]
                # print(target_data["h_algo_0_reps"], target_data["h_algo_0_reps"].shape)

                training_data = moleculize(target_data[training_name])
                print(training_data.shape, training_data.size)
                xtr_algo_0_d = e_train.transform(training_data)
                # plot_tsne(xtr_algo_0_d, np.ones((xtr_algo_0_d.shape[0])))

                print(target_rep.shape, target_rep.size)
                xta_algo_0_d = e_train.transform(target_rep.sum(axis=0).reshape(1, -1))
                # plot_tsne(xta_algo_0_d, np.full((xta_algo_0_d.shape[0]), fill_value=2))
                x_all = np.concatenate((x_qm7, xtr_algo_0_d, xta_algo_0_d), axis=0)
                y_all = np.concatenate(
                    (
                        np.zeros((x_qm7.shape[0])),
                        np.ones((xtr_algo_0_d.shape[0])),
                        np.full((xta_algo_0_d.shape[0]), fill_value=2),
                    ),
                    axis=0,
                )
                pickle.dump(e_train, open(f"{training_name}_global.sav", "wb"))
                plot_tsne(x_all, y_all, target_name, f"{training_name}_global")

        return None
    else:
        qm7_reps = np.concatenate(qm7_reps, axis=0)
        qm7_ncharges = np.concatenate(qm7_ncharges, axis=0)
        #print(
        #    qm7_reps.shape,
        #    qm7_reps.size,
        #    qm7_ncharges.shape,
        #    qm7_ncharges.size,
        #    qm7_ncharges[0:10],
        #)

        # V2, good for C
        #tsne = TSNE(
        #    perplexity=10,
        #    metric="euclidean",
        #    n_jobs=-1,
        #    random_state=42,
        #    initialization="random",
        #    verbose=True,
        #    early_exaggeration_iter=50,
        #)
        # early_exaggeration_iter=10,
        # n_iter=10,

        # V3
        tsne = TSNE(
            perplexity=2,
            metric="euclidean",
            n_jobs=-1,
            random_state=42,
            initialization="random",
            verbose=True,
            early_exaggeration_iter=10,
        )

        selected_atom = 7
        qm7_reps = qm7_reps[np.where(qm7_ncharges == selected_atom)[0]]
        print("After filter:", qm7_reps.shape, qm7_reps.size)

        sav_path = f"{selected_atom}_local_v2.sav"
        if os.path.isfile(sav_path):
            print(f"loading from {sav_path}")
            with open(sav_path, "rb") as f:
                e_train = pickle.load(f)
        else:
            print(f"fitting and saving to {sav_path}")
            e_train = tsne.fit(qm7_reps)
            with open(sav_path, "wb") as f:
                pickle.dump(e_train, f)
        print()
        x_sav_path = f"qm7_{selected_atom}_local_v2.sav"
        if os.path.isfile(x_sav_path):
            print(f"loading from {x_sav_path}")
            with open(x_sav_path, "rb") as f:
                x_qm7 = pickle.load(f)
        else:
            print(f"fitting and saving to {x_sav_path}")
            x_qm7 = e_train.transform(qm7_reps)
            with open(x_sav_path, "wb") as f:
                pickle.dump(x_qm7, f)
        print()


        training_set_names = [
            "h_algo_0_reps",
            "h_algo_1_reps",
            "h_random_reps",
            "h_cur_reps",
            "h_sml_reps",
            "h_fps_reps",
        ]
        training_set_name_ncharges = [
            "h_algo_0_ncharges",
            "h_algo_1_ncharges",
            "h_random_ncharges",
            "h_cur_ncharges",
            "h_sml_ncharges",
            "h_fps_ncharges",
        ]
        for t, target_data in enumerate(targets_data):
            for training_name, ncharges_name in zip(
                training_set_names, training_set_name_ncharges
            ):
                print(f'{training_name=}')
                target_rep = target_data["target_rep"]
                target_ncharges = target_data["target_ncharges"]
                print(target_rep.shape, target_rep.size, target_ncharges[0:10])
                target_rep = target_rep[np.where(target_ncharges == selected_atom)[0]]
                print("After filter:", target_rep.shape, target_rep.size)
                if target_rep.size == 0:
                    continue
                if not isinstance(target_rep, np.ndarray):
                    target_rep.reshape(1, -1)
                target_name = target_data["target_name"]
                print(f'{target_name=}')
                xta_algo_0_d = e_train.transform(target_rep)

                training_data = np.concatenate(target_data[training_name], axis=0)
                training_ncharges = np.concatenate(target_data[ncharges_name], axis=0)
                print(f'{training_data.shape=} {training_data.size=}')
                training_data = training_data[
                    np.where(training_ncharges == selected_atom)[0]
                ]
                print("After filter:", training_data.shape, training_data.size)
                if training_data.size == 0:
                    continue
                xtr_algo_0_d = e_train.transform(training_data)
                x_all = np.concatenate((x_qm7, xtr_algo_0_d, xta_algo_0_d), axis=0)
                y_all = np.concatenate(
                    (
                        np.zeros((x_qm7.shape[0])),
                        np.ones((xtr_algo_0_d.shape[0])),
                        np.full((xta_algo_0_d.shape[0]), fill_value=2),
                    ),
                    axis=0,
                )
                plot_tsne(
                    x_all,
                    y_all,
                    target_name,
                    f"{training_name}_{selected_atom}_local_v3",
                )
                print()
            print()
        return None

    # Flatten the database to get the total number of atoms and their atom types
    # flattened_Xs = [
    #    np.array(atom_features) for sublist in mixed_Xs for atom_features in sublist
    # ]
    # flattened_atomtypes = [
    #    atomtype for sublist in atomtypes_database for atomtype in sublist
    # ]
    # total_database_atoms = len(flattened_Xs)


def moleculize(target_data_list):
    # print(target_data_list[0].shape)
    training_data = np.zeros((len(target_data_list), target_data_list[0].shape[1]))
    for k, molecule in enumerate(target_data_list):
        training_data[k, :] = molecule.sum(axis=0)
    return training_data


def atomize_training_set(target_data_list):
    # print(target_data_list[0].shape)
    training_data = np.zeros((len(target_data_list), target_data_list[0].shape[1]))
    for k, molecule in enumerate(target_data_list):
        for i, atom in enumerate(molecule):
            training_data[k, :] = molecule.sum(axis=0)
    return training_data


def distance_distribution_plots(targets_data, database="drugs", option="all"):
    """
    Plot all distances observed for a set of target molecules and their associated subsets of molecules in a single plot.

    Parameters:
    targets_data (list of dict): List of dictionaries where each dictionary contains the following keys:
                                 - 'target_rep': The feature vectors for the target atoms.
                                 - 'target_ncharges': The atom types for the target atoms.
                                 - 'h_algo_0_reps': The feature vectors for the ILP(p=0) subset.
                                 - 'h_algo_0_ncharges': The atom types for the ILP(p=0) subset.
                                 - 'h_algo_1_reps': The feature vectors for the ILP(p=1) subset.
                                 - 'h_algo_1_ncharges': The atom types for the ILP(p=1) subset.
                                 - 'h_random_reps': The feature vectors for the random subset.
                                 - 'h_random_ncharges': The atom types for the random subset.
                                 - 'h_cur_reps': The feature vectors for the CUR subset.
                                 - 'h_cur_ncharges': The atom types for the CUR subset.
                                 - 'h_sml_reps': The feature vectors for the SML subset.
                                 - 'h_sml_ncharges': The atom types for the SML subset.
                                 - 'h_fps_reps': The feature vectors for the FPS subset.
                                 - 'h_fps_ncharges': The atom types for the FPS subset.
                                 - 'target_name': A name or identifier for the target molecule.
    option (str): Option for computing distances ('all' or 'min').

    Returns:
    None
    """

    # Initialize lists to store the distances
    all_algo_0_d = []
    all_algo_1_d = []
    all_random_d = []
    all_cur_d = []
    all_sml_d = []
    all_fps_d = []

    for t, target_data in enumerate(targets_data):
        target_rep = target_data["target_rep"]
        target_ncharges = target_data["target_ncharges"]
        target_name = target_data["target_name"]

        # Compute distances for each subset
        algo_0_d = compute_pairwise_distances_database(
            target_data["h_algo_0_reps"],
            target_data["h_algo_0_ncharges"],
            option=option,
        )
        algo_1_d = compute_pairwise_distances_database(
            target_data["h_algo_1_reps"],
            target_data["h_algo_1_ncharges"],
            option=option,
        )
        random_d = compute_pairwise_distances_database(
            target_data["h_random_reps"],
            target_data["h_random_ncharges"],
            option=option,
        )
        cur_d = compute_pairwise_distances_database(
            target_data["h_cur_reps"], target_data["h_cur_ncharges"], option=option
        )
        sml_d = compute_pairwise_distances_database(
            target_data["h_sml_reps"], target_data["h_sml_ncharges"], option=option
        )
        fps_d = compute_pairwise_distances_database(
            target_data["h_fps_reps"], target_data["h_fps_ncharges"], option=option
        )

        # Append the distances to the lists
        # all_algo_0_d.extend(algo_0_d)
        # all_algo_1_d.extend(algo_1_d)
        # all_random_d.extend(random_d)
        # all_cur_d.extend(cur_d)
        # all_sml_d.extend(sml_d)
        # all_fps_d.extend(fps_d)

        # Convert lists to numpy arrays for plotting
        all_algo_0_d = filter(np.array(algo_0_d))
        all_algo_1_d = filter(np.array(algo_1_d))
        all_random_d = filter(np.array(random_d))
        all_cur_d = filter(np.array(cur_d))
        all_sml_d = filter(np.array(sml_d))
        all_fps_d = filter(np.array(fps_d))

        # want to normalise by number of atoms somehow

        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 6))
        sns.kdeplot(
            all_algo_0_d, label="ILP(p=0)", color=colors[0], linestyle="dashed", ax=ax
        )
        sns.kdeplot(all_algo_1_d, label="ILP(p=1)", color=colors[1], ax=ax)
        sns.kdeplot(all_random_d, label="random", color=colors[2], ax=ax)
        sns.kdeplot(all_cur_d, label="CUR", color=colors[3], ax=ax)
        sns.kdeplot(all_sml_d, label="SML", color=colors[4], ax=ax)
        sns.kdeplot(all_fps_d, label="FPS", color=colors[5], ax=ax)
        ax.set_xlabel("Euclidean distance to target atoms")
        ax.set_ylabel("Count")
        ax.legend()
        plt.tight_layout()
        plt.savefig(
            f"interpret_figs/dist_training_{option}_{database}_{target_name}.pdf"
        )
        # plt.show()
    return


def combined_distance_plot(targets_data, database="drugs", option="all"):
    """
    Plot all distances observed for a set of target molecules and their associated subsets of molecules in a single plot.

    Parameters:
    targets_data (list of dict): List of dictionaries where each dictionary contains the following keys:
                                 - 'target_rep': The feature vectors for the target atoms.
                                 - 'target_ncharges': The atom types for the target atoms.
                                 - 'h_algo_0_reps': The feature vectors for the ILP(p=0) subset.
                                 - 'h_algo_0_ncharges': The atom types for the ILP(p=0) subset.
                                 - 'h_algo_1_reps': The feature vectors for the ILP(p=1) subset.
                                 - 'h_algo_1_ncharges': The atom types for the ILP(p=1) subset.
                                 - 'h_random_reps': The feature vectors for the random subset.
                                 - 'h_random_ncharges': The atom types for the random subset.
                                 - 'h_cur_reps': The feature vectors for the CUR subset.
                                 - 'h_cur_ncharges': The atom types for the CUR subset.
                                 - 'h_sml_reps': The feature vectors for the SML subset.
                                 - 'h_sml_ncharges': The atom types for the SML subset.
                                 - 'h_fps_reps': The feature vectors for the FPS subset.
                                 - 'h_fps_ncharges': The atom types for the FPS subset.
                                 - 'target_name': A name or identifier for the target molecule.
    option (str): Option for computing distances ('all' or 'min' or 'max').

    Returns:
    None
    """

    # Initialize lists to store the distances
    all_algo_0_d = []
    all_algo_1_d = []
    all_random_d = []
    all_cur_d = []
    all_sml_d = []
    all_fps_d = []

    for target_data in targets_data:
        target_rep = target_data["target_rep"]
        target_ncharges = target_data["target_ncharges"]

        # Compute distances for each subset
        algo_0_d = np.concatenate(
            compute_pairwise_distances(
                target_data["h_algo_0_reps"],
                target_rep,
                target_data["h_algo_0_ncharges"],
                target_ncharges,
                option=option,
            ),
            axis=0,
        )
        algo_1_d = np.concatenate(
            compute_pairwise_distances(
                target_data["h_algo_1_reps"],
                target_rep,
                target_data["h_algo_1_ncharges"],
                target_ncharges,
                option=option,
            ),
            axis=0,
        )
        random_d = np.concatenate(
            compute_pairwise_distances(
                target_data["h_random_reps"],
                target_rep,
                target_data["h_random_ncharges"],
                target_ncharges,
                option=option,
            ),
            axis=0,
        )
        cur_d = np.concatenate(
            compute_pairwise_distances(
                target_data["h_cur_reps"],
                target_rep,
                target_data["h_cur_ncharges"],
                target_ncharges,
                option=option,
            ),
            axis=0,
        )
        sml_d = np.concatenate(
            compute_pairwise_distances(
                target_data["h_sml_reps"],
                target_rep,
                target_data["h_sml_ncharges"],
                target_ncharges,
                option=option,
            ),
            axis=0,
        )
        fps_d = np.concatenate(
            compute_pairwise_distances(
                target_data["h_fps_reps"],
                target_rep,
                target_data["h_fps_ncharges"],
                target_ncharges,
                option=option,
            ),
            axis=0,
        )

        # Append the distances to the lists
        all_algo_0_d.extend(algo_0_d)
        all_algo_1_d.extend(algo_1_d)
        all_random_d.extend(random_d)
        all_cur_d.extend(cur_d)
        all_sml_d.extend(sml_d)
        all_fps_d.extend(fps_d)

    # Convert lists to numpy arrays for plotting
    all_algo_0_d = filter(np.array(all_algo_0_d))
    all_algo_1_d = filter(np.array(all_algo_1_d))
    all_random_d = filter(np.array(all_random_d))
    all_cur_d = filter(np.array(all_cur_d))
    all_sml_d = filter(np.array(all_sml_d))
    all_fps_d = filter(np.array(all_fps_d))

    # want to normalise by number of atoms somehow

    # Plotting all distances in a single plot
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 6))
    sns.kdeplot(
        all_algo_0_d, label="ILP(p=0)", color=colors[0], linestyle="dashed", ax=ax
    )
    sns.kdeplot(all_algo_1_d, label="ILP(p=1)", color=colors[1], ax=ax)
    sns.kdeplot(all_random_d, label="random", color=colors[2], ax=ax)
    sns.kdeplot(all_cur_d, label="CUR", color=colors[3], ax=ax)
    sns.kdeplot(all_sml_d, label="SML", color=colors[4], ax=ax)
    sns.kdeplot(all_fps_d, label="FPS", color=colors[5], ax=ax)
    ax.set_xlabel("Euclidean distance to target atoms")
    ax.set_ylabel("Count")
    ax.legend()

    plt.tight_layout()
    plt.savefig(f"interpret_figs/combined_dist_targets_{option}_{database}.pdf")
    # plt.show()
    return


def similarity_plot(
    algo_0_reps,
    algo_0_ncharges,
    algo_1_reps,
    algo_1_ncharges,
    fps_reps,
    fps_ncharges,
    sml_reps,
    sml_ncharges,
    cur_reps,
    cur_ncharges,
    random_reps,
    random_ncharges,
    target_rep,
    target_ncharges,
    database="drugs",
):
    # get global similarity
    algo_0_K = local_global_sim(
        algo_0_reps, target_rep, algo_0_ncharges, target_ncharges
    )
    algo_1_K = local_global_sim(
        algo_1_reps, target_rep, algo_1_ncharges, target_ncharges
    )
    fps_K = local_global_sim(fps_reps, target_rep, fps_ncharges, target_ncharges)
    sml_K = local_global_sim(sml_reps, target_rep, sml_ncharges, target_ncharges)
    cur_K = local_global_sim(cur_reps, target_rep, cur_ncharges, target_ncharges)
    random_K = local_global_sim(
        random_reps, target_rep, random_ncharges, target_ncharges
    )

    K_min = np.min(np.concatenate((algo_0_K, algo_1_K, fps_K, sml_K, cur_K, random_K)))
    K_max = np.max(np.concatenate((algo_0_K, algo_1_K, fps_K, sml_K, cur_K, random_K)))

    fig, ax = plt.subplots(nrows=1, ncols=1)
    sns.kdeplot(
        algo_0_K, label="ILP(p=0)", alpha=0.8, color=colors[0], linestyle="dashed"
    )
    sns.kdeplot(algo_1_K, label="ILP(p=1)", alpha=0.8, color=colors[1])
    sns.kdeplot(fps_K, label="FPS", alpha=0.8, color=colors[2])
    sns.kdeplot(cur_K, label="CUR", alpha=0.8, color=colors[3])
    sns.kdeplot(sml_K, label="SML", alpha=0.8, color=colors[4])
    sns.kdeplot(random_K, label="Random", alpha=0.8, color=colors[5])
    ax.set_xlabel("Local kernel similarity")
    plt.legend()
    plt.savefig(f"interpret_figs/{target}_hists_sim_{database}.pdf")
    plt.show()
    return


# NOW ONLY PLOTTING DISTS BUT OTHER FUNCS ARE ALSO AVAILABLE (SIMILARITIES ETC)
args = parse_args()
target = args.target
database = args.database
colors = ["tab:blue", "tab:blue", "tab:purple", "tab:red", "tab:orange", "tab:green"]

if database != "qm7":
    df = pd.read_csv("targets/energies.csv")
else:
    df = pd.read_csv("qm7/energies.csv")
if database == "drugs":
    targets = [
        "apixaban",
        "imatinib",
        "oseltamivir",
        "oxycodone",
        "pemetrexed",
        "penicillin",
        "pregabalin",
        "salbutamol",
        "sildenafil",
        "troglitazone",
    ]

elif database == "qm9":
    targets = [
        "121259",
        "12351",
        "35811",
        "85759",
        "96295",
        "5696",
        "31476",
        "55607",
        "68076",
        "120425",
    ]

elif database == "qm7":
    targets = [
        "qm7_1251",
        "qm7_3576",
        "qm7_6163",
        "qm7_1513",
        "qm7_1246",
        "qm7_2161",
        "qm7_6118",
        "qm7_5245",
        "qm7_5107",
        "qm7_3037",
    ]

else:
    raise NotImplementedError("only qm7, qm9 and drugs not implemented")

if target != "all":
    if database != "qm7":
        target_name = target + ".xyz"
    else:
        target_name = target
    y_target = float(df[df["file"] == target_name]["energy / Ha"])

    target_rep, target_ncharges, h_target_rep, h_target_ncharges = load_reps_target(
        target
    )
    (
        algo_1_ncharges,
        algo_1_reps,
        sizes_algo_1,
        h_algo_1_ncharges,
        h_algo_1_reps,
        algo_0_ncharges,
        algo_0_reps,
        sizes_algo_0,
        h_algo_0_ncharges,
        h_algo_0_reps,
        cur_ncharges,
        cur_reps,
        sizes_cur,
        h_cur_ncharges,
        h_cur_reps,
        fps_ncharges,
        fps_reps,
        sizes_fps,
        h_fps_ncharges,
        h_fps_reps,
        sml_ncharges,
        sml_reps,
        sizes_sml,
        h_sml_ncharges,
        h_sml_reps,
        random_ncharges,
        random_reps,
        sizes_random,
        h_random_ncharges,
        h_random_reps,
    ) = load_qm7(target)

    if args.size:
        size_plot(
            sizes_algo_0, sizes_algo_1, sizes_random, sizes_cur, sizes_sml, sizes_fps
        )

    distance_plot(
        h_algo_0_reps,
        h_algo_0_ncharges,
        h_algo_1_reps,
        h_algo_1_ncharges,
        h_random_reps,
        h_random_ncharges,
        h_cur_reps,
        h_cur_ncharges,
        h_sml_reps,
        h_sml_ncharges,
        h_fps_reps,
        h_fps_ncharges,
        h_target_rep,
        h_target_ncharges,
        database=database,
    )

else:
    targets_data = []
    sizes_targets_data = []
    for target in targets:
        if database != "qm7":
            target_name = target + ".xyz"
        else:
            target_name = target
        y_target = float(df[df["file"] == target_name]["energy / Ha"])

        target_rep, target_ncharges, h_target_rep, h_target_ncharges = load_reps_target(
            target
        )
        (
            algo_1_ncharges,
            algo_1_reps,
            sizes_algo_1,
            h_algo_1_ncharges,
            h_algo_1_reps,
            algo_1_idxs,
            algo_0_ncharges,
            algo_0_reps,
            sizes_algo_0,
            h_algo_0_ncharges,
            h_algo_0_reps,
            algo_0_idxs,
            cur_ncharges,
            cur_reps,
            sizes_cur,
            h_cur_ncharges,
            h_cur_reps,
            cur_idxs,
            fps_ncharges,
            fps_reps,
            sizes_fps,
            h_fps_ncharges,
            h_fps_reps,
            fps_idxs,
            sml_ncharges,
            sml_reps,
            sizes_sml,
            h_sml_ncharges,
            h_sml_reps,
            sml_idxs,
            random_ncharges,
            random_reps,
            sizes_random,
            h_random_ncharges,
            h_random_reps,
            random_idxs,
        ), qm7_ncharges, qm7_reps = load_qm7(target)

        if args.size:
            sizes_targets_data.append(
                {
                    "sizes_algo_0": sizes_algo_0,
                    "sizes_algo_1": sizes_algo_1,
                    "sizes_random": sizes_random,
                    "sizes_cur": sizes_cur,
                    "sizes_sml": sizes_sml,
                    "sizes_fps": sizes_fps,
                    "target_name": target,
                }
            )

        targets_data.append(
            {
                "target_rep": h_target_rep,
                "target_ncharges": h_target_ncharges,
                "h_algo_0_reps": h_algo_0_reps,
                "h_algo_0_ncharges": h_algo_0_ncharges,
                "h_algo_0_idxs": algo_0_idxs,
                "h_algo_1_reps": h_algo_1_reps,
                "h_algo_1_ncharges": h_algo_1_ncharges,
                "h_algo_1_idxs": algo_1_idxs,
                "h_random_reps": h_random_reps,
                "h_random_ncharges": h_random_ncharges,
                "h_random_idxs": random_idxs,
                "h_cur_reps": h_cur_reps,
                "h_cur_ncharges": h_cur_ncharges,
                "h_cur_idxs": cur_idxs,
                "h_sml_reps": h_sml_reps,
                "h_sml_ncharges": h_sml_ncharges,
                "h_sml_idxs": sml_idxs,
                "h_fps_reps": h_fps_reps,
                "h_fps_ncharges": h_fps_ncharges,
                "h_fps_idxs": fps_idxs,
                "target_name": target,
            }
        )
    # if args.size:
    #    combined_size_plot_stacked(sizes_targets_data, database=database)
    #combined_distance_plot(targets_data, database=database)
    # combined_distance_plot(targets_data, database=database, option="min")
    # combined_distance_plot(targets_data, database=database, option="max")
    # combined_distance_plot(targets_data, database=database, option="delta")
    # distance_distribution_plots(targets_data, database=database)
    # tsne_plots(qm7_reps, qm7_ncharges, targets_data, global_rep=True)
    print('tsne')
    tsne_plots(qm7_reps, qm7_ncharges, targets_data, global_rep=False)

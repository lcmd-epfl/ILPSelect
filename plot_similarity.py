import numpy as np
#import qml
from sklearn.metrics import pairwise_distances
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
#from sklearn.manifold import TSNE
from matplotlib.colors import Normalize
np.random.seed(20)
plt.rcParams["figure.figsize"] = (7,4.8)
from scripts.kernels import get_local_kernel
import seaborn as sns
import argparse as ap
import pandas as pd

pt = {"C":6, "N":7, "O":8, "S":16, "F":9, "H":1}

def parse_args():
    parser = ap.ArgumentParser()
    parser.add_argument('-p', '--preg', action='store_true')
    parser.add_argument('-s', '--size_plot', action='store_true')
    parser.add_argument('-dp', '--dissim_plot', action='store_true')
    parser.add_argument('-d', '--distances_plot', action='store_true')
    parser.add_argument('-m', '--min', action='store_true')
    parser.add_argument('-sp', '--sim_plot', action='store_true')
    parser.add_argument('-pca', '--pca_plot', action='store_true')
    args = parser.parse_args()
    return args

def global_similarity(Xs, X_target, sigma=10):
    # maybe need to modify to return max similarity
    X_target = X_target.reshape(1, -1)
    D = pairwise_distances(X_target, Xs)
    K = np.exp(-D / (2*sigma**2)).reshape(-1)
    return K


def compute_pairwise_distances(mixed_Xs, X_target, atomtypes_database, atomtypes_target, option='all'):
    """
    Compute the squared Euclidean distances between each target atom and all atoms in the database.

    Parameters:
    mixed_Xs (list of list of lists): The database of atomic environments with mixed dimensions.
    X_target (numpy.ndarray): The feature vectors for the target atoms with dimensions (target_dim, feature_size).
    atomtypes_database (list of lists): Atom types for each atom in the database with mixed dimensions.
    atomtypes_target (list): Atom types for each atom in the target molecule.
    option (str): 'all' to return all distances, 'min' to return only the minimum distances.

    Returns:
    numpy.ndarray: A 2D array of pairwise distances with dimensions (target_dim, total_database_atoms) if option is 'all'.
                   A 2D array of minimum distances with dimensions (target_dim, n_samples) if option is 'min'.
    """
    target_dim, target_feature_size = X_target.shape
    n_samples = len(mixed_Xs)

    # Flatten the database to get the total number of atoms and their atom types
    flattened_Xs = [np.array(atom_features) for sublist in mixed_Xs for atom_features in sublist]
    flattened_atomtypes = [atomtype for sublist in atomtypes_database for atomtype in sublist]
    total_database_atoms = len(flattened_Xs)

    if option == 'all':
        # Initialize the distance matrix for 'all' option
        distances = np.full((target_dim, total_database_atoms), np.inf)

        # Compute distances
        for i in range(target_dim):
            for j, (db_atom_features, db_atomtype) in enumerate(zip(flattened_Xs, flattened_atomtypes)):
                if atomtypes_target[i] == db_atomtype:
                    distances[i, j] = np.linalg.norm(X_target[i] - db_atom_features) ** 2

        return distances

    elif option == 'min':
        # Initialize the distance matrix for 'min' option
        min_distances = np.zeros((target_dim, n_samples))

        # Compute minimum distances
        for i in range(target_dim):
            for sample_index, (sublist, sublist_atomtypes) in enumerate(zip(mixed_Xs, atomtypes_database)):
                min_distance = np.inf
                for atom_features, atomtype in zip(sublist, sublist_atomtypes):
                    if atomtypes_target[i] == atomtype:
                        db_atom_features = np.array(atom_features)
                        distance = np.linalg.norm(X_target[i] - db_atom_features) ** 2
                        if distance < min_distance:
                            min_distance = distance
                if min_distance == np.inf:
                    min_distance = np.nan  # Handle cases where no matching atom type is found
                min_distances[i, sample_index] = min_distance

        return min_distances

def local_global_sim(Xs, X_target, Qs, Q_target, sigma=1):
    K = get_local_kernel(Xs, np.array([X_target]), Qs, np.array([Q_target]), sigma=sigma)
    # flatten is assuming there was a single target X_target
    return K.flatten()

def get_global_rep(ncharges, reps):
    h_reps = []
    h_ncharges = []
    for i, mol_ncharges in enumerate(ncharges):
        h_filter = np.where(mol_ncharges != 1)
        X = reps[i][h_filter]
        h_ncharges.append(mol_ncharges[h_filter])
        h_reps.append(X)

    return h_reps, h_ncharges

def get_molecule_sizes(ncharges_list, heavy=True):
    sizes = []
    for ncharges in ncharges_list:
        ncharges = [x for x in ncharges if x!=1]
        size = len(ncharges)
        sizes.append(size)
    return sizes

def get_qm7_from_indices(index_list):
    df_qm7 = pd.read_csv('qm7/energies.csv')
    ys = []
    for file in index_list:
        y = float(df_qm7[df_qm7['file'] == file]['energy / Ha'])
        ys.append(y)
    if len(ys) != len(index_list):
        raise ValueError(f'some indices are wrong since {len(ys)=}, {len(index_list)=}')
    return np.array(ys)

args = parse_args()
if args.preg:
    local_path = 'local_preg'
    global_path = 'npys_preg'
    target = 'pregabalin'
else:
    local_path = 'local_npys'
    global_path= 'npys'
    target = 'sildenafil'

df = pd.read_csv('targets/energies.csv')
y_target = float(df[df['file'] == target+'.xyz']['energy / Ha'])

# LOAD GLOBAL FROM FILES
target_rep = np.load(f'{global_path}/target_rep.npy', allow_pickle=True)
algo_reps_0 = np.load(f'{global_path}/algo_reps_0.npy', allow_pickle=True)
algo_reps_1 = np.load(f'{global_path}/algo_reps_1.npy', allow_pickle=True)
sml_reps = np.load(f'{global_path}/sml_reps.npy', allow_pickle=True)

fps_reps = np.load(f'{global_path}/fps_reps.npy', allow_pickle=True)
cur_reps = np.load(f'{global_path}/cur_reps.npy', allow_pickle=True)
random_reps = np.load(f'{global_path}/random_reps.npy', allow_pickle=True)

# LOAD LOCAL FROM FILES
l_target_rep = np.load(f'{local_path}/target_rep.npy', allow_pickle=True)
l_target_ncharges = np.load(f'{local_path}/target_ncharges.npy', allow_pickle=True)
h_filter = np.where(l_target_ncharges != 1)
l_h_target_rep = l_target_rep[h_filter]
l_h_target_ncharges = l_target_ncharges[h_filter]

l_algo_reps_0 = np.load(f'{local_path}/algo_reps_0.npy', allow_pickle=True)
l_algo_ncharges_0 = np.load(f'{local_path}/algo_ncharges_0.npy', allow_pickle=True)
l_h_algo_reps_0, h_ncharges_algo_0 = get_global_rep(l_algo_ncharges_0, l_algo_reps_0)
sizes_algo_0 = get_molecule_sizes(l_algo_ncharges_0)
l_algo_0_indices = np.load(f'data/algo_FCHL_qm7_{target}_0.npy')
y_algo_0 = get_qm7_from_indices(l_algo_0_indices)

l_algo_reps_1 = np.load(f'{local_path}/algo_reps_1.npy', allow_pickle=True)
l_algo_ncharges_1 = np.load(f'{local_path}/algo_ncharges_1.npy', allow_pickle=True)
l_h_algo_reps_1, h_ncharges_algo_1 = get_global_rep(l_algo_ncharges_1, l_algo_reps_1)
sizes_algo_1 = get_molecule_sizes(l_algo_ncharges_1)
l_algo_1_indices = np.load(f'data/algo_FCHL_qm7_{target}_1.npy')
y_algo_1 = get_qm7_from_indices(l_algo_1_indices)

l_sml_reps = np.load(f'{local_path}/sml_reps.npy', allow_pickle=True)
l_sml_ncharges = np.load(f'{local_path}/sml_ncharges.npy', allow_pickle=True)
l_h_sml_reps, ncharges_sml = get_global_rep(l_sml_ncharges, l_sml_reps)
sizes_sml = get_molecule_sizes(l_sml_ncharges)
y_sml_indices = np.load(f'data/sml_FCHL_qm7_{target}.npy')
y_sml = get_qm7_from_indices(y_sml_indices)

l_fps_reps = np.load(f'{local_path}/fps_reps.npy', allow_pickle=True)
l_fps_ncharges = np.load(f'{local_path}/fps_ncharges.npy', allow_pickle=True)
l_h_fps_reps, ncharges_fps = get_global_rep(l_fps_ncharges, l_fps_reps)
sizes_fps = get_molecule_sizes(l_fps_ncharges)
fps_indices = np.load('data/fps_FCHL_qm7.npy')
y_fps = get_qm7_from_indices(fps_indices)

l_cur_reps = np.load(f'{local_path}/cur_reps.npy', allow_pickle=True)
l_cur_ncharges = np.load(f'{local_path}/cur_ncharges.npy', allow_pickle=True)
l_h_cur_reps, ncharges_cur = get_global_rep(l_cur_ncharges, l_cur_reps)
sizes_cur = get_molecule_sizes(l_cur_ncharges)
cur_indices = np.load('data/cur_FCHL_qm7.npy')
y_cur = get_qm7_from_indices(cur_indices)

l_random_reps = np.load(f'{local_path}/random_reps.npy', allow_pickle=True)
l_random_ncharges = np.load(f'{local_path}/random_ncharges.npy', allow_pickle=True)
l_h_random_reps, ncharges_random = get_global_rep(l_random_ncharges, l_random_reps)
sizes_random = get_molecule_sizes(l_random_ncharges)
random_indices = np.load('data/random_files.npy')
random_indices = [x.split('/')[1].split('.xyz')[0] for x in random_indices]
y_random = get_qm7_from_indices(random_indices)

colors = ['tab:blue', 'tab:blue', 'tab:purple', 'tab:red', 'tab:orange', 'tab:green']

############
# SIZE PLOT #
if args.size_plot:
    fig, ax = plt.subplots(nrows=1, ncols=1)
    sns.kdeplot(sizes_algo_0, label='ILP(p=0)', color=colors[0], linestyle='dashed', alpha=0.8)
    sns.kdeplot(sizes_algo_1, label='ILP(p=1)', color=colors[1], alpha=0.8)
    sns.kdeplot(sizes_random, label='random', color=colors[2], alpha=0.8)
    sns.kdeplot(sizes_cur, label='CUR', color=colors[3], alpha=0.8)
    sns.kdeplot(sizes_sml, label='SML', color=colors[4], alpha=0.8)
    sns.kdeplot(sizes_fps, label='FPS', color=colors[5], alpha=0.8)
    ax.set_xlabel("Number of heavy atoms")
    ax.set_ylabel("Count")
    plt.legend()
    if args.preg:
        plt.savefig('interpret_figs/preg_sizes.pdf')
    else:
        plt.savefig('interpret_figs/sizes.pdf')
    plt.show()
###########

# get global similarity
algo_0_K = local_global_sim(l_algo_reps_0, l_target_rep, l_algo_ncharges_0, l_target_ncharges)
algo_1_K = local_global_sim(l_algo_reps_1, l_target_rep, l_algo_ncharges_1, l_target_ncharges)
fps_K = local_global_sim(l_fps_reps, l_target_rep, l_fps_ncharges, l_target_ncharges)
sml_K = local_global_sim(l_sml_reps, l_target_rep, l_sml_ncharges, l_target_ncharges)
cur_K = local_global_sim(l_cur_reps, l_target_rep, l_cur_ncharges, l_target_ncharges)
random_K = local_global_sim(l_random_reps, l_target_rep, l_random_ncharges, l_target_ncharges)

K_min = np.min(np.concatenate((algo_0_K, algo_1_K, fps_K, sml_K, cur_K, random_K)))
K_max = np.max(np.concatenate((algo_0_K, algo_1_K, fps_K, sml_K, cur_K, random_K)))
#################
# DISSIM PLOT
#################
if args.dissim_plot:
    fig, ax = plt.subplots(nrows=1, ncols=1)
    # difference in property (y) vs kernel (x)
    ax.set_ylabel('$y - y_{T}$')
    ax.set_xlabel('Normalized Similarity kernel')
    ax.scatter(algo_0_K / K_max, y_algo_0 - y_target, label='ILP(p=0)', color='grey')
    ax.scatter(algo_1_K / K_max, y_algo_1 - y_target, label='ILP(p=1)', color=colors[1], alpha=0.6)
    ax.scatter(random_K / K_max, y_random - y_target, label='Random', color=colors[2], alpha=0.6)
    ax.scatter(cur_K / K_max, y_cur - y_target, label='CUR', color=colors[3], alpha=0.6)
    ax.scatter(sml_K / K_max, y_sml - y_target, label='SML', color=colors[4], alpha=0.6)
    ax.scatter(fps_K / K_max, y_fps - y_target, label='FPS', color=colors[5], alpha=0.6)
    plt.legend()
    plt.tight_layout()

    if not args.preg:
        plt.savefig('interpret_figs/dissim_plot.pdf')
    else:
        plt.savefig(f'interpret_figs/dissim_plot_{target}.pdf')
    plt.show()
#################

# distance
if args.min:
    option = 'min'
else:
    option = 'all'
algo_O_d = np.concatenate(compute_pairwise_distances(l_h_algo_reps_0, l_h_target_rep, h_ncharges_algo_0, l_h_target_ncharges, option=option), axis=0)
algo_1_d = np.concatenate(compute_pairwise_distances(l_h_algo_reps_1, l_h_target_rep, h_ncharges_algo_1, l_h_target_ncharges, option=option), axis=0)
random_d = np.concatenate(compute_pairwise_distances(l_h_random_reps, l_h_target_rep, ncharges_random, l_h_target_ncharges, option=option), axis=0)
cur_d = np.concatenate(compute_pairwise_distances(l_h_cur_reps, l_h_target_rep, ncharges_cur, l_h_target_ncharges, option=option), axis=0)
sml_d = np.concatenate(compute_pairwise_distances(l_h_sml_reps, l_h_target_rep, ncharges_sml, l_h_target_ncharges, option=option), axis=0)
fps_d = np.concatenate(compute_pairwise_distances(l_h_fps_reps, l_h_target_rep, ncharges_fps, l_h_target_ncharges, option=option), axis=0)

#################
# DISTANCES PLOT
#################
if args.distances_plot:
    fig, ax = plt.subplots(nrows=1, ncols=1)
    sns.kdeplot(algo_O_d, label='ILP(p=0)', color=colors[0], linestyle='dashed')
    sns.kdeplot(algo_1_d, label='ILP(p=1)', color=colors[1])
    sns.kdeplot(random_d, label='random', color=colors[2])
    sns.kdeplot(cur_d, label='CUR', color=colors[3])
    sns.kdeplot(sml_d, label='SML', color=colors[4])
    sns.kdeplot(fps_d, label='FPS', color=colors[5])
    ax.set_xlim(0, 13)
    ax.set_xlabel('Euclidean distance to target atoms')
    plt.legend()
    if args.preg:
        plt.savefig(f'interpret_figs/preg_dist_targets_{option}.pdf')
    else:
        plt.savefig(f'interpret_figs/dist_targets_{option}.pdf')
    plt.show()
##################

#################
# LOCAL SIM HIST
#################
if args.sim_plot:
    fig, ax  = plt.subplots(nrows=1, ncols=1)
    sns.kdeplot(algo_0_K, label='ILP(p=0', alpha=0.8, color=colors[0], linestyle='dashed')
    sns.kdeplot(algo_1_K, label='ILP(p=1)', alpha=0.8, color=colors[1])
    sns.kdeplot(fps_K, label='FPS', alpha=0.8, color=colors[2])
    sns.kdeplot(sml_K, label='SML', alpha=0.8, color=colors[3])
    sns.kdeplot(cur_K, label='CUR', alpha=0.8, color=colors[4])
    sns.kdeplot(random_K, label='Random', alpha=0.8, color=colors[5])
    ax.set_xlabel("Local kernel similarity")
    plt.legend()
    if args.preg:
        plt.savefig('interpret_figs/hists_preg_sim.pdf')
    else:
        plt.savefig("interpret_figs/hists_sim.pdf")
    plt.show()
################

#########################
# PCA ##################
#######################
if args.pca_plot:
    pca = PCA(n_components=2)
    pca_algo_0 = pca.fit_transform(algo_reps_0)
    pca_algo_1 = pca.fit_transform(algo_reps_1)

    pca_fps = pca.fit_transform(fps_reps)
    pca_sml = pca.fit_transform(sml_reps)
    pca_cur = pca.fit_transform(cur_reps)

    pca_random = pca.fit_transform(random_reps)

    norm = Normalize(vmin=K_min, vmax=K_max)

    fig, axes = plt.subplots(nrows=2, ncols=3)

    ax = axes[0,0]
    ax.set_title("ILP(p=1)")
    ax.scatter(pca_algo_1[:,0], pca_algo_1[:,1], c=algo_1_K / K_max, cmap='viridis')
    ax.axis('off')

    ax = axes[0,1]
    ax.set_title("ILP(p=0)")
    ax.scatter(pca_algo_0[:,0], pca_algo_0[:,1], c=algo_0_K / K_max, cmap='viridis')
    ax.axis('off')

    ax = axes[0,2]
    ax.set_title("Random")
    ax.scatter(pca_random[:,0],pca_random[:,1], c=random_K / K_max, cmap='viridis')
    ax.axis('off')

    ax = axes[1,0]
    ax.set_title("FPS")
    sc = ax.scatter(pca_fps[:,0], pca_fps[:,1], c=fps_K / K_max, cmap='viridis')
    ax.axis('off')

    ax = axes[1,2]
    ax.set_title("SML")
    ax.scatter(pca_sml[:,0], pca_sml[:,1], c=sml_K / K_max, cmap='viridis')
    ax.axis('off')

    ax = axes[1,1]
    ax.set_title("CUR")
    ax.scatter(pca_cur[:,0], pca_cur[:,1], c=cur_K / K_max, cmap='viridis')
    ax.axis('off')

    fig.subplots_adjust(right=0.85)  # adjust the right margin to make space for the colorbar
    cbar_ax = fig.add_axes([0.87, 0.15, 0.03, 0.7])  # [left, bottom, width, height]
    cbar = fig.colorbar(sc, cax=cbar_ax)
    cbar.set_label('Local kernel similarity to target', fontsize=11)
    plt.tight_layout(rect=[0, 0, 0.85, 1])

    if args.preg:
        plt.savefig("interpret_figs/local_sim_newK_preg.pdf")
    else:
        plt.savefig("interpret_figs/local_sim_newK.pdf")
    plt.show()
#######################
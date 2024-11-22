import os
import pickle
import argparse as ap
import pandas as pd
import numpy as np
from openTSNE import TSNE
from plot_similarity import get_molecule_sizes, load_qm7, load_reps_target


def parse_args():
    parser = ap.ArgumentParser()
    parser.add_argument("-d", "--database", default="drugs")
    parser.add_argument("--tsne_atom", type=int, default=16)
    args = parser.parse_args()
    return args


def get_data_for_tsne_plots(
    qm7_reps,
    qm7_ncharges,
    targets_data,
    selected_atom=6,
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

    Returns:
    None
    """
    qm7_reps_full = qm7_reps

    # a bit dumb
    qm7_ncharges_full = np.zeros_like(qm7_reps_full, dtype=int)[:,:,0]
    for i, ncharges in enumerate(qm7_ncharges):
        qm7_ncharges_full[i,:len(ncharges)] = ncharges

    qm7_reps = np.concatenate(qm7_reps_full, axis=0)
    qm7_ncharges = np.concatenate(qm7_ncharges_full, axis=0)

    perplexity = {6: 500,
                  16: 4,
                  8: 80,
                  7: 90,
                  }

    tsne = TSNE(
        perplexity=perplexity[selected_atom],
        metric="euclidean",
        n_jobs=-1,
        random_state=42,
        initialization="random",
        verbose=True,
        early_exaggeration_iter=50,
    )

    qm7_reps = qm7_reps[np.where(qm7_ncharges == selected_atom)[0]]
    print("After filter:", qm7_reps.shape, qm7_reps.size)

    sav_path = f"tsne_cache/{selected_atom}_local_perp{perplexity[selected_atom]}.sav"
    x_sav_path = f"tsne_cache/qm7_{selected_atom}_local_perp{perplexity[selected_atom]}.sav"
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
    if os.path.isfile(x_sav_path):
        print(f"loading from {x_sav_path}")
        with open(x_sav_path, "rb") as f:
            x_qm7 = pickle.load(f)
    else:
        print(f"transforming and saving to {x_sav_path}")
        x_qm7 = e_train.transform(qm7_reps)
        with open(x_sav_path, "wb") as f:
            pickle.dump(x_qm7, f)
    print()


    algos = ["algo_0", "algo_1", "random", "cur", "sml", "fps"]

    for target_data in targets_data:
        for algo in algos:
            print(f'{algo=}')
            idxs_name = f'h_{algo}_idxs'
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

            # also dumb but i don't know how to do it beautifully
            atom_mol_idx = np.full((qm7_ncharges_full.shape[::-1]), np.arange(len(qm7_ncharges_full))).T
            atom_mol_idx = np.concatenate(atom_mol_idx, axis=0)[np.where(qm7_ncharges == selected_atom)]
            selected_atom_idx = []
            for i in target_data[idxs_name]:
                selected_atom_idx.extend(np.where(atom_mol_idx==i)[0])
            selected_atom_idx = np.array(selected_atom_idx)

            if selected_atom_idx.size == 0:
                continue
            xtr_algo_0_d = x_qm7[selected_atom_idx]

            x_qm7_rest = x_qm7[np.setdiff1d(np.arange(len(x_qm7)), selected_atom_idx)]

            x_all = np.concatenate((x_qm7_rest, xtr_algo_0_d, xta_algo_0_d), axis=0)
            y_all = np.concatenate(
                (
                    np.zeros((x_qm7_rest.shape[0])),
                    np.ones((xtr_algo_0_d.shape[0])),
                    np.full((xta_algo_0_d.shape[0]), fill_value=2),
                ),
                axis=0,
            )
            np.savez(f"interpret_figs/tsne/tsne_{target_name}_{selected_atom}_perp{perplexity[selected_atom]}_{algo}",
                     x=x_all, y=y_all)
            print()
        print()

    return



args = parse_args()
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

targets_data = []
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
get_data_for_tsne_plots(qm7_reps, qm7_ncharges, targets_data, selected_atom=args.tsne_atom)

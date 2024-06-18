import numpy as np

def read_xyz(xyz_path):
    with open(xyz_path, 'r') as f:
        lines = f.readlines()
    lines = [l.strip() for l in lines]
    nat = int(lines[0])

    ncharges = []
    coords = []
    for line in lines[2:]:
        at, x, y, z = line.split()
        ncharges.append(at)
        coords.append([x,y,z])

    assert len(ncharges) == len(coords)

    return np.array(ncharges, dtype=str), np.array(coords, dtype=float)

def get_global_FCHL(ncharges, coords, elements=None, max_natoms=None):
    if not elements:
        elements = np.unique(ncharges)
    if not max_natoms:
        max_natoms = len(ncharges)

    rep = qml.representations.generate_fchl_acsf(
                ncharges,
                coords,
                elements=elements,
                gradients=False,
                pad=max_natoms,
            )
    print(rep.shape)
    g_rep = np.sum(rep, axis=0)
    print(g_rep.shape)
    return g_rep

def xyz_to_rep(xyz_path, elements=None, max_natoms=None):
    ncharges, coords = read_xyz(xyz_path)
    g_rep = get_global_FCHL(ncharges, coords,
                            elements=elements, max_natoms=max_natoms)
    return g_rep

# GLOBAL similarity to the target molecule
target_xyz = 'targets/sildenafil.xyz'
target_rep = xyz_to_rep(target_xyz)
# FCHL19 NEEDS TO WORK ON CLUSTER
exit()

algo = np.load('data/algo_FCHL_qm7_sildenafil_1.npy')
algo_xyzs = [f'qm7/{x}.xyz' for x in algo]
cur = np.load('data/cur_FCHL_qm7.npy')
cur_xyzs = [f'qm7/{x}.xyz' for x in cur]
fps = np.load('data/fps_FCHL_qm7.npy')
fps_xyzs = [f'qm7/{x}.xyz' for x in fps]
sml = np.load('data/sml_FCHL_qm7_sildenafil.npy')
sml_xyzs = [f'qm7/{x}.xyz' for x in sml]

# get coords, ncharges from xyzs
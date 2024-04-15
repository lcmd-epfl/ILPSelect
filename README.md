## Requirements

The code was run on Python 3.10.12. The following modules are required: gurobipy, numpy, pandas, qml, sklearn, skmatter, plotly, kaleido, pickle, json, os (last three are installed by default).
```
python3 -m pip install gurobipy numpy pandas qml scikit-learn skmatter plotly kaleido
```

Gurobi is used to solve integer linear programs (FPS and ILP). A license is required. Academic licenses are available, and clusters need special licenses.
The following environment variable should point towards the license file in case gurobipy cannot find it on its own.
```
export GRB_LICENSE_FILE=/ssoft/spack/external/gurobi/gurobi.lic
``` 

## First Run

- to use `algo_model` and `algo_subset`, the molekuehl folder should be added to the python path.
Add this line at the end of your .bashrc file to add it systematically:
```
export PYTHONPATH="${PYTHONPATH}:/home/haeberle/molekuehl/"
```

- create folder models, rankings, solutions, and learning_curves. The .mps files it contains are large and thus in the .gitignore.
```
mkdir models rankings solutions learning_curves
```

- verify that the folder `qm7` exists, and that it contains the energies described in an `energies.csv` file (with columns `file` and `energy / Ha`).

The `main.py` file runs everything based on a Python config file. The default config files `config.py` used by default when running `main.py` with no argument.
In order to use custom config `config-foo.py`, use the command `python3 main.py "config-foo"`.

## Walkthrough of `main.py`

The `main.py` combines all files of folder scripts to do the following.

- Read target names from config file. The corresponding `{target_name}.xyz` files should be present in the folder `targets`.
- Read the config script for the following parameters: database (qm7 for now), representation (FCHL), algorithm-specific parameters, learning curve parameters, ...
- Generate the representations if not present (with convention `{rep}\_{target}.npz` and `{rep}\_{database}.npz`) and save to folder `data`. The database must be in the `{database}` folder.
The file `scripts/generate.py` is responsible for this step.
- Compute fragment subsets using different techniques (indices of database)
	- Subset selection by ILP (named `algo` in the code):
		- Generate model and write it to `models` folder OR read it and modify its parameters if possible (simple penalty change for example).
		The file `scripts/algo_model.py` is responsible for this step, and is based off of the file `script/fragments.py`.
		- Solve model and output subset to folder `rankings` with prefix `algo_`, and solution of ILP to folder `solutions`.
		The file `scripts/algo_subset.py` is responsible for this step.
	- Subset selection by SML:
		- Output subset to `rankings` folder with prefix `sml_`.
		The file `scripts/sml_subset.py` is responsible for this step.
	- Subset selection by CUR:
		- Output subset to `rankings` with prefix `cur_`.
		The file `scripts/cur_subset.py` is responsible for this step.
	- Subset selection by FPS:
		- Output subset to `rankings` with prefix `fps_`.
		The file `scripts/fps_subset.py` is responsible for this step.
- Compute the learning curve of each subset and save to folder `learning_curves`.
The file `scripts/learning_curves.py` is responsible for this step.
- Draw the learning curves and save to folder `plots`.
The file `scripts/plots.py` is responsible for this step.
- The timings of each step are saved in a dump file in the `run` folder. An example file `dump-template.csv` can be found in the folder.

## Running on clusters

The folder `run` contains a `main.run` file which describes how the scripts were ran on the JED cluster.
An example output file `slurm.out` is included.

## Adding targets, databases, representations

### Targets

Add a `{target_name}.xyz` file to the folder `targets`.
Add a corresponding entry with the associated energy in the `energies.csv` file in the same folder.

### Databases

Create a `{database}` folder, which contains the energies described in an `energies.csv` file (with columns `file` and `energy / Ha`).
One may add a column `atomization energy / Ha`. 
See the `qm9/generate.py` script and the `cluster/scripts/generate_qm9.py` file in the master branch for an example of a qm9 implementation from a master file.

### Representation

Modify accordingly the file `scripts/generate.py`. Currently the `get_representations` function asserts that FCHL is used.

## TODO

- Run example
- Complete the python mkdocs
- Implement qm9 database. The only thing currently missing is some pruning of the database because Gurobi uses too much memory (even on clusters). 
- Implement other representations than FCHL. Not a priority but should not be too difficult (`representation` is already a parameter).

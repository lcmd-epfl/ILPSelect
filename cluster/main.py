# %%
%reload_ext autoreload
%autoreload 2
from config import config

# read target names, database
target_names = config["target_names"]
print(f"Read {len(target_names)} target(s): {target_names}")

database = config["database"]

# %%
# generate representations
from scripts.generate import generate_database, generate_targets

representation = config["representation"]
repository_folder = config["repository_folder"]
current_folder = config["current_folder"]

if config["remove_target_from_database"]:
    fragment_to_remove = 1

generate_database(database, representation, repository_folder, )

generate_targets(target_names, representation, current_folder)

size_subset = config["learning_curve_ticks"][-1]

# %%
# generate sml subset

from scripts.sml_subset import sml_subset

sml_subset(
    parent_folder=current_folder,
    database=database,
    targets=target_names,
    representation=representation,
    N=size_subset,
    remove_target_from_database=config["remove_target_from_database"],
)

# %%
# generate algo model

from scripts.algo_model import algo_model

algo_model(
    repository_path=repository_folder,
    database=database,
    targets=target_names,
    representation=representation,
    config=config,
)

# %% generate algo subset
from scripts.algo_subset import algo_subset

algo_subset(
    repository_path=repository_folder,
    database=database,
    targets=target_names,
    representation=representation,
    N=size_subset,
    config=config,
)
# %%
# generate learning curves
from scripts.learning_curves import learning_curves, learning_curves_random

algorithms = ["fragments", "sml"]
learning_curves(
    repository_path=repository_folder,
    database=database,
    targets=target_names,
    representation=representation,
    config=config,
    algorithms=algorithms,
)
#%%
CV = 1
learning_curves_random(
    parent_directory=repository_folder,
    database=database,
    targets=target_names,
    representation=representation,
    config=config,
    CV=CV,
)

# %%
# draw learning curves
from scripts.plots import plots

plots(
    parent_directory=current_folder,
    database=database,
    targets=target_names,
    representation=representation,
    config=config,
    algorithms=algorithms,
)

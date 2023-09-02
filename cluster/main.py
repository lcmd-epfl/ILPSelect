# %%
import time

from config import config

# read config
target_names = config["target_names"]
print(f"Read {len(target_names)} target(s): {target_names}")

database = config["database"]

representation = config["representation"]
repository_folder = config["repository_folder"]
current_folder = config["current_folder"]

size_subset = config["learning_curve_ticks"][-1]

algorithms = ["fragments", "sml"]

dump = {"num_targets": len(target_names)}

# %%
# generate representations
from scripts.generate import generate_database, generate_targets

if config["generate_database"]:
    t = time.time()
    generate_database(database, representation, repository_folder)
    t = time.time() - t
    dump["time_generate_database"] = t

if config["generate_targets"]:
    t = time.time()
    generate_targets(target_names, representation, current_folder)
    t = time.time() - t
    dump["time_generate_targets"] = t


# %%
# generate sml subset

from scripts.sml_subset import sml_subset

if config["sml_subset"]:
    t = time.time()
    sml_subset(
        parent_folder=current_folder,
        database=database,
        targets=target_names,
        representation=representation,
        N=size_subset,
        remove_target_from_database=config["remove_target_from_database"],
    )
    t = time.time() - t
    dump["time_sml_subset"] = t

# %%
# generate algo model

from scripts.algo_model import algo_model

if config["algo_model"]:
    t = time.time()
    algo_model(
        repository_path=repository_folder,
        database=database,
        targets=target_names,
        representation=representation,
        config=config,
    )
    t = time.time() - t
    dump["time_algo_model"] = t

# %% generate algo subset
from scripts.algo_subset import algo_subset

if config["algo_subset"]:
    t = time.time()
    algo_subset(
        repository_path=repository_folder,
        database=database,
        targets=target_names,
        representation=representation,
        N=size_subset,
        config=config,
    )
    t = time.time() - t
    dump["time_algo_subset"] = t
# %%
# generate learning curves
from scripts.learning_curves import learning_curves

if config["learning_curves"]:
    t = time.time()
    learning_curves(
        repository_path=repository_folder,
        database=database,
        targets=target_names,
        representation=representation,
        config=config,
        algorithms=algorithms,
    )
    t = time.time() - t
    dump["time_learning_curves"] = t
# %%
from scripts.learning_curves import learning_curves_random

CV = 5
if config["learning_curves_random"]:
    t = time.time()
    learning_curves_random(
        repository_path=repository_folder,
        database=database,
        targets=target_names,
        representation=representation,
        config=config,
        CV=CV,
        add_onto_old=True,
    )
    t = time.time() - t
    dump["time_learning_curves_random"] = t

# %%
# draw learning curves
from scripts.plots import plots

if config["plots"]:
    t = time.time()
    plots(
        parent_directory=current_folder,
        database=database,
        targets=target_names,
        representation=representation,
        config=config,
        algorithms=algorithms,
    )
    t = time.time() - t
    dump["time_plots"] = t

# %%
import json
from datetime import datetime

time = datetime.now().strftime("%Y-%m-%d")
DUMP_PATH = f"run/dump-{time}.json"
with open(DUMP_PATH, "w") as f:
    json.dump(dump, f, indent=2)

print(f"Dumped timings to {DUMP_PATH}")

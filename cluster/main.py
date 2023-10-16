# %%
import time
from datetime import datetime

import pandas as pd

from config import config

# read config
target_names = config["target_names"]
print(f"Read {len(target_names)} target(s): {target_names}")

database = config["database"]

representation = config["representation"]
repository_folder = config["repository_folder"]
current_folder = config["current_folder"]

size_subset = config["learning_curve_ticks"][-1]

# timings dump
current_time = datetime.now().strftime("%Y-%m-%d")
DUMP_PATH = f"{repository_folder}cluster/run/dump-{current_time}.csv"

dump = pd.DataFrame(
    {"Property": ["num_targets", "targets"], "Value": [len(target_names), target_names]}
)
dump.to_csv(DUMP_PATH)

# %%
# generate representations
from scripts.generate import generate_database, generate_targets

if config["generate_database"]:
    t = time.time()
    generate_database(
        database,
        representation,
        repository_folder,
        targets=target_names,
        in_database=config["in_database"],
    )
    t = time.time() - t
    dump = pd.concat([dump, pd.DataFrame({"Property": ["time_generate_database"], "Value": [t]})])
    dump.to_csv(DUMP_PATH)

if config["generate_targets"]:
    t = time.time()
    generate_targets(
        targets=target_names,
        representation=representation,
        repository_path=repository_folder,
        database=database,
        in_database=config["in_database"],
    )
    t = time.time() - t
    dump = pd.concat([dump, pd.DataFrame({"Property": ["time_generate_targets"], "Value": [t]})])
    dump.to_csv(DUMP_PATH)


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
        in_database=config["in_database"],
    )
    t = time.time() - t
    dump = pd.concat([dump, pd.DataFrame({"Property": ["time_sml_subset"], "Value": [t]})])
    dump.to_csv(DUMP_PATH)
# %%
# generate fps, cur subset

from scripts.fps_cur_subset import cur_subset, fps_subset

if config["cur_subset"]:
    t = time.time()
    cur_subset(
        parent_folder=current_folder,
        database=database,
        targets=target_names,
        representation=representation,
        N=size_subset,
        in_database=config["in_database"],
    )
    t = time.time() - t
    dump = pd.concat([dump, pd.DataFrame({"Property": ["time_cur_subset"], "Value": [t]})])
    dump.to_csv(DUMP_PATH)

if config["fps_subset"]:
    t = time.time()
    fps_subset(
        parent_folder=current_folder,
        database=database,
        targets=target_names,
        representation=representation,
        N=size_subset,
        in_database=config["in_database"],
    )
    t = time.time() - t
    dump = pd.concat([dump, pd.DataFrame({"Property": ["time_fps_subset"], "Value": [t]})])
    dump.to_csv(DUMP_PATH)

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
    dump = pd.concat([dump, pd.DataFrame({"Property": ["time_algo_model"], "Value": [t]})])
    dump.to_csv(DUMP_PATH)

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
    dump = pd.concat([dump, pd.DataFrame({"Property": ["time_algo_subset"], "Value": [t]})])
    dump.to_csv(DUMP_PATH)

# %%
# generate learning curves
from scripts.learning_curves import learning_curves

if len([e for e in config["learning_curves"] if e != "random"]) != 0:
    t = time.time()
    learning_curves(
        repository_path=repository_folder,
        database=database,
        targets=target_names,
        representation=representation,
        config=config,
        curves=config["learning_curves"],  # TODO: add fps when implemented
    )
    t = time.time() - t
    dump = pd.concat([dump, pd.DataFrame({"Property": ["time_learning_curves"], "Value": [t]})])
    dump.to_csv(DUMP_PATH)
# %%
from scripts.learning_curves import learning_curves_random

if "random" in config["learning_curves"]:
    t = time.time()
    learning_curves_random(
        repository_path=repository_folder,
        database=database,
        targets=target_names,
        representation=representation,
        config=config,
        CV=config["CV"],
        add_onto_old=False,
    )
    t = time.time() - t
    dump = pd.concat(
        [dump, pd.DataFrame({"Property": ["time_learning_curves_random"], "Value": [t]})]
    )
    dump.to_csv(DUMP_PATH)

# %%
# draw learning curves
from scripts.plots import plots_average, plots_individual

if len(config["plots_individual"]) != 0:
    t = time.time()
    plots_individual(
        parent_directory=current_folder,
        database=database,
        targets=target_names,
        representation=representation,
        pen=config["penalty"],
        learning_curve_ticks=config["learning_curve_ticks"],
        curves=config["plots_individual"],  # TODO: add fps when implemented
    )
    t = time.time() - t
    dump = pd.concat([dump, pd.DataFrame({"Property": ["time_plots_individual"], "Value": [t]})])
    dump.to_csv(DUMP_PATH)

if len(config["plots_average"]) != 0:
    t = time.time()
    plots_average(
        parent_directory=current_folder,
        database=database,
        targets=config["plot_average_target_names"],
        representation=representation,
        pen=config["penalty"],
        learning_curve_ticks=config["learning_curve_ticks"],
        curves=config["plots_average"],
    )
    t = time.time() - t
    dump = pd.concat([dump, pd.DataFrame({"Property": ["time_plots_average"], "Value": [t]})])
    dump.to_csv(DUMP_PATH)


# %%

"""
Main file agregator that reads a python config file and runs each script accordingly.

Running with no argument takes the template config.py.
    >>> python3 main.py
Running with an argument searches for the config name.
    >>> python3 main.py "config-qm7drugs"
"""

# %%
import sys
import time
from datetime import datetime

import pandas as pd

# read config
if len(sys.argv) > 1:
    # argument gives special config name, eg config-drugs or config-qm7fragments, ..
    config = __import__(sys.argv[1]).config
else:
    # default file config.py
    from config import config

target_names = config["target_names"]
config_name = config["config_name"]
print(f"Config name {config_name}")
print(f"Read {len(target_names)} target(s): {target_names}")

database = config["database"]

representation = config["representation"]
repository_folder = config["repository_folder"]
current_folder = config["repository_folder"]

size_subset = config["learning_curve_ticks"][-1]

# timings dump
current_time = datetime.now().strftime("%Y-%m-%d")
DUMP_PATH = f"{repository_folder}run/dump-{config_name}-{current_time}.csv"

dump = pd.DataFrame(
    {"Property": ["num_targets", "targets"], "Value": [len(target_names), target_names]}
)
dump.to_csv(DUMP_PATH)


# concat and save function
def add_onto_and_save(df, prop, value):
    df = pd.concat([df, pd.DataFrame({"Property": [prop], "Value": [value]})])
    global DUMP_PATH
    df.to_csv(DUMP_PATH)
    return df

# %%
# generate representations

if config["generate_database"]:
    from scripts.generate import generate_database
    timer = time.time()
    generate_database(config)
    timer = time.time() - timer
    dump = add_onto_and_save(dump, "time_generate_database", timer)

if config["generate_targets"]:
    from scripts.generate import generate_targets
    timer = time.time()
    generate_targets(config)
    timer = time.time() - timer
    dump = add_onto_and_save(dump, "time_generate_targets", timer)


# %%
# generate fps, cur subset


if config["cur_subset"]:
    from scripts.cur_subset import cur_subset
    timer = time.time()
    cur_subset(config)
    timer = time.time() - timer
    dump = add_onto_and_save(dump, "time_cur_subset", timer)


if config["fps_subset"]:
    from scripts.fps_subset import fps_subset
    timer = time.time()
    fps_subset(config)
    timer = time.time() - timer
    dump = add_onto_and_save(dump, "time_fps_subset", timer)

# %%
# generate sml subset


if config["sml_subset"]:
    from scripts.sml_subset import sml_subset
    timer = time.time()
    sml_subset(config)
    timer = time.time() - timer
    dump = add_onto_and_save(dump, "time_sml_subset", timer)

# %%
# generate algo model


if config["algo_model"]:
    from scripts.algo_model import algo_model
    timer = time.time()
    algo_model(config)
    timer = time.time() - timer
    dump = add_onto_and_save(dump, "time_algo_model", timer)

# %% generate algo subset

if config["algo_subset"]:
    from scripts.algo_subset import algo_subset
    timer = time.time()
    algo_subset(config)
    timer = time.time() - timer
    dump = add_onto_and_save(dump, "time_algo_subset", timer)

# %%
# generate learning curves

no_random_curves = [e for e in config["learning_curves"] if e != "random"]

if len(no_random_curves) != 0:
    from scripts.learning_curves import learning_curves
    timer = time.time()
    learning_curves(config)
    timer = time.time() - timer
    dump = add_onto_and_save(dump, "time_learning_curves", timer)
# %%

if "random" in config["learning_curves"]:
    from scripts.learning_curves import learning_curves
    timer = time.time()
    learning_curves(config, random=True)
    timer = time.time() - timer
    dump = add_onto_and_save(dump, "time_learning_curves_random", timer)

# %%
# draw learning curves

if len(config["plots_individual"]) != 0:
    from scripts.plots import plots_individual
    timer = time.time()
    plots_individual(config)
    timer = time.time() - timer
    dump = add_onto_and_save(dump, "time_plots_individual", timer)

if len(config["plots_average"]) != 0:
    from scripts.plots import plots_average
    timer = time.time()
    plots_average(config)
    timer = time.time() - timer
    dump = add_onto_and_save(dump, "time_plots_average", timer)


# %%

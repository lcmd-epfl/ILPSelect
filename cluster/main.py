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
current_folder = config["current_folder"]

size_subset = config["learning_curve_ticks"][-1]

# timings dump
current_time = datetime.now().strftime("%Y-%m-%d")
DUMP_PATH = f"{repository_folder}cluster/run/dump-{config_name}-{current_time}.csv"

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
from scripts.generate import generate_database, generate_targets

if config["generate_database"]:
    t = time.time()
    generate_database(config)
    t = time.time() - t
    dump = add_onto_and_save(dump, "time_generate_database", t)

if config["generate_targets"]:
    t = time.time()
    generate_targets(config)
    t = time.time() - t
    dump = add_onto_and_save(dump, "time_generate_targets", t)


# %%
# generate sml subset

from scripts.sml_subset import sml_subset

if config["sml_subset"]:
    t = time.time()
    sml_subset(config)
    t = time.time() - t
    dump = add_onto_and_save(dump, "time_sml_subset", t)
# %%
# generate fps, cur subset

from scripts.cur_subset import cur_subset

if config["cur_subset"]:
    t = time.time()
    cur_subset(config)
    t = time.time() - t
    dump = add_onto_and_save(dump, "time_cur_subset", t)

from scripts.fps_subset import fps_subset

if config["fps_subset"]:
    t = time.time()
    fps_subset(config)
    t = time.time() - t
    dump = add_onto_and_save(dump, "time_fps_subset", t)

# %%
# generate algo model

from scripts.algo_model import algo_model

if config["algo_model"]:
    t = time.time()
    algo_model(config)
    t = time.time() - t
    dump = add_onto_and_save(dump, "time_algo_model", t)

# %% generate algo subset
from scripts.algo_subset import algo_subset

if config["algo_subset"]:
    t = time.time()
    algo_subset(config)
    t = time.time() - t
    dump = add_onto_and_save(dump, "time_algo_subset", t)

# %%
# generate learning curves
from scripts.learning_curves import learning_curves

no_random_curves = [e for e in config["learning_curves"] if e != "random"]

if len(no_random_curves) != 0:
    t = time.time()
    learning_curves(config)
    t = time.time() - t
    dump = add_onto_and_save(dump, "time_learning_curves", t)
# %%
from scripts.learning_curves import learning_curves_random

if "random" in config["learning_curves"]:
    t = time.time()
    learning_curves_random(config, add_onto_old=False)
    t = time.time() - t
    dump = add_onto_and_save(dump, "time_learning_curves_random", t)

# %%
# draw learning curves
from scripts.plots import plots_average, plots_individual

if len(config["plots_individual"]) != 0:
    t = time.time()
    plots_individual(config)
    t = time.time() - t
    dump = add_onto_and_save(dump, "time_plots_individual", t)

if len(config["plots_average"]) != 0:
    t = time.time()
    plots_average(config)
    t = time.time() - t
    dump = add_onto_and_save(dump, "time_plots_average", t)


# %%

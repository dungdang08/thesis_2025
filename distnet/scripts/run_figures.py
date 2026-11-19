# Adapted from DistNet (https://github.com/KEggensperger/DistNet),
# Eggensperger, Lindauer & Hoos, IJCAI 2018. Modified by Hoang Dung Dang (thesis student);
# see ATTRIBUTION.md for the list of changes.

import sys

import numpy as np

sys.path.append("../")
from helper import load_data, data_source_release


SCENARIOS = [
    "clasp_factoring",
    "saps-CVVAR",
    "spear_qcp",
    "yalsat_qcp",
    "spear_swgcp",
    "yalsat_swgcp",
    "lpg-zeno",
]
FOLDS = list(range(10))

# 3) Load data
sc_dict = data_source_release.get_sc_dict()
data_dir = data_source_release.get_data_dir()
with open("training_data_num.txt", "w") as f_out:
    for scenario in SCENARIOS:
        if scenario not in sc_dict.keys():
            raise ValueError(f"Scenario {scenario} not found in sc_dict. "
                                f"Available scenarios: {list(sc_dict.keys())}")
        runtimes, features, _ = load_data.get_data(
            scenario=scenario, 
            data_dir=data_dir,
            sc_dict=sc_dict, 
            retrieve=sc_dict[scenario]['use'])
        features = np.array(features)
        runtimes = np.array(runtimes)
        f_out.write(f"\n\nScenario: {scenario}\n")
        f_out.write(f"runtimes: {runtimes.shape}, features: {features.shape}\n")

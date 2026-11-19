# Adapted from DistNet (https://github.com/KEggensperger/DistNet),
# Eggensperger, Lindauer & Hoos, IJCAI 2018. Modified by Hoang Dung Dang (thesis student);
# see ATTRIBUTION.md for the list of changes.

import os
import sys

import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize

from . import preprocess

def read_performance(algo, dim, budget, runs, agregate_runs=False, log_target=False):
    log_flag = "_log" if log_target else ""
    agregate_runs_flag = "_per_run" if not agregate_runs else ""
    
    SC = f"{algo}_{dim}D_{budget}"
    # Load performance data
    fl_name = f"../data/processed_data/performance_data_regression{log_flag}/{SC}{agregate_runs_flag}.csv"
    try:
        performance_df = pd.read_csv(fl_name, index_col=0)
    except FileNotFoundError:
        print(f"File {fl_name} not found. Skipping {SC}.")
        return [], []
    except pd.errors.EmptyDataError:
        print(f"File {fl_name} is empty. Skipping {SC}.")
        return [], []
    except pd.errors.ParserError:
        print(f"File {fl_name} could not be parsed. Skipping {SC}.")
        return [], []
    except Exception as e:
        print(f"An unexpected error occurred while reading {fl_name}: {e}")
        return [], []
    
    performance_df["group"] = performance_df.index.str.split("_").str[:2].str.join("_")
    target_precs = performance_df.groupby("group")["target"].apply(list)
    # Convert to numpy array
    data = np.array(target_precs.tolist())
    inst_ls = performance_df["group"].unique().tolist()

    return data, inst_ls


def load_features(dim):
    feat_dict = dict()
    fl_name = f"../data/processed_data/landscape_data/ELA_LHS_{dim}D_1000n_100seed_reduced.csv"
    try:
        feat_df = pd.read_csv(fl_name, index_col=0)
    except FileNotFoundError:
        print(f"File {fl_name} not found.")
        return feat_dict
    except pd.errors.EmptyDataError:
        print(f"File {fl_name} is empty.")
        return feat_dict
    except pd.errors.ParserError:
        print(f"File {fl_name} could not be parsed.")
        return feat_dict
    except Exception as e:
        print(f"An unexpected error occurred while reading {fl_name}: {e}")
        return feat_dict
    for i in range(feat_df.shape[0]):
        key = feat_df.index[i]
        val = feat_df.iloc[i, :].values.astype(float).tolist()
        feat_dict[key] = val
    return feat_dict


def get_data(algo, dim, budget, runs, agregate_runs=False, log_target=False):
    target_precs, inst_ls = read_performance(algo, dim, budget, runs, agregate_runs, log_target)
    feat_dict = load_features(dim=dim)
    
    SC = f"{algo}_{dim}D_{budget}"
    print(SC)
    features = list()
    for i in inst_ls:
        if i not in feat_dict:
            print(f"Warning: {i} not found in feature dictionary.")
            continue
        features.append(feat_dict[i])
    features = np.array(features)
    target_precs = np.array(target_precs)
    print(features.shape, target_precs.shape)

    # Remove all instances with constant features
    target_precs, features, inst_ls = preprocess.remove_constant_instances(
            runningtimes=target_precs, features=features, sat_ls=None)
    
    # Remove all instances with target precisions close/equal to zeros
    target_precs, features, inst_ls = preprocess.remove_zeros(
        runningtimes=target_precs, features=features, sat_ls=None)
    

    return target_precs, features, inst_ls
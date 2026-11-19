# Adapted from DistNet (https://github.com/KEggensperger/DistNet),
# Eggensperger, Lindauer & Hoos, IJCAI 2018. Modified by Hoang Dung Dang (thesis student);
# see ATTRIBUTION.md for the list of changes.

import argparse
import os
import pickle
import sys

import numpy as np
from sklearn.model_selection import KFold

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..', 'third_party', 'xgboostlss'))

from xgboostlss.model import *
from xgboostlss.distributions.LogNormal import LogNormal
from xgboostlss.distributions.Weibull import Weibull
from xgboostlss.distributions.Exponential import Exponential
from xgboostlss.distributions.InverseGaussian import InverseGaussian

sys.path.append("../")
from helper import load_data, preprocess, data_source_release


def main():
    sc_dict = data_source_release.get_sc_dict()
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", dest="model", required=True,
                        choices=(
                                 "invgauss_xgb.floc",
                                 "lognormal_xgb.floc",
                                 "expon_xgb.floc",
                                 "weibull_xgb.floc",
                                ))
    parser.add_argument("--scenario", dest="scenario", required=True,
                        choices=sc_dict.keys())
    parser.add_argument("--num_train_samples", dest="num_train_samples",
                        default=100, type=int)
    parser.add_argument("--fold", dest="fold", required=True, type=int)
    parser.add_argument("--save", dest="save", required=True)
    parser.add_argument("--seed", dest="seed", required=False, default=1,
                        type=int)
    parser.add_argument("--wclim", dest="wclim", required=False, default=60*59,
                        type=int)
    parser.add_argument("--neurons", dest="neurons", required=False,
                        default=16, type=int)
    parser.add_argument("--layer", dest="layer", required=False, default=2,
                        type=int)
    parser.add_argument("--epochs", dest="epochs", required=False, default=1000,
                        type=int)
    args = parser.parse_args()
    # 1) Assertions
    assert 0 <= args.fold <= 9

    # 2) Sort out model and task
    model_name, task = args.model.split(".")
    save_path = os.path.join(args.save, "%s.%s.%s.%d.%d.pkl" % (args.scenario,
                                                                task, model_name,
                                                                args.fold,
                                                                args.seed))
    if args.num_train_samples != 100:
        save_path += "_%d" % args.num_train_samples
    add_info = {"task": task, "scenario": args.scenario,
                "fold": args.fold, "model": model_name, "loaded": False,
                "num_train_samples": args.num_train_samples,
                "seed": args.seed}

    # Maybe load configuration from file
    config_fp = os.path.join(args.save, "config",
                             "%s.%s.%s.cfg" % (args.scenario, model_name, task))
    print("CHECKING %s" % config_fp)
    if os.path.exists(config_fp):
        with open(config_fp, "rb") as fh:
            config = pickle.load(fh)
            add_info["loaded"] = True
        print("Successfully loaded config %s" % config_fp)
    else:
        config = None

    # 3) Load data
    sc_dict = data_source_release.get_sc_dict()
    data_dir = data_source_release.get_data_dir()

    runtimes, features, sat_ls = load_data.\
        get_data(scenario=args.scenario, data_dir=data_dir,
                 sc_dict=sc_dict, retrieve=sc_dict[args.scenario]['use'])

    features = np.array(features)
    runtimes = np.array(runtimes)
    
    # Get CV splits
    print(runtimes.shape, features.shape)
    idx = list(range(runtimes.shape[0]))
    kf = KFold(n_splits=10, shuffle=True, random_state=0)
    cntr = -1
    for train, valid in kf.split(idx):
        # Reset seed for every instance
        np.random.seed(2)
        cntr += 1
        if cntr != args.fold:
            continue

        X_train = features[train, :]
        X_valid = features[valid, :]

        y_train = runtimes[train]
        y_valid = runtimes[valid]

        X_train, X_valid = preprocess.preprocess_features(X_train, X_valid,
                                                          scal="meanstd")

        print("Evaluating %s, %s, %s on %s, %s" % (args.scenario, model_name,
                                                   task, str(X_train.shape),
                                                   str(y_train.shape)))

        X_trn_flat = np.concatenate(
            [[x for i in range(100)] for x in X_train])
        X_vld_flat = np.concatenate(
            [[x for i in range(100)] for x in X_valid])
        y_trn_flat = y_train.flatten().reshape([-1, 1])
        y_vld_flat = y_valid.flatten().reshape([-1, 1])

        # Unfold data
        subset_idx = list(range(100))
        if args.num_train_samples != 100:
            print("Cut data down to %d samples with seed %d" %
                  (args.num_train_samples, args.seed))
            rs = np.random.RandomState(args.seed)
            rs.shuffle(subset_idx)
            subset_idx = subset_idx[:args.num_train_samples]
            print(subset_idx)

            # Only shorten data used for training
            X_trn_flat = np.concatenate(
                [[x for i in range(args.num_train_samples)] for x in X_train])
            y_train = y_train[:, subset_idx]
            y_trn_flat = y_train.flatten().reshape([-1, 1])

            X_vld_flat = np.concatenate(
                [[x for i in range(args.num_train_samples)] for x in X_valid])
            y_valid = y_valid[:, subset_idx]
            y_vld_flat = y_valid.flatten().reshape([-1, 1])

        # Min/Max Scale runnningtimes
        y_max_ = np.max(y_trn_flat)
        y_min_ = 0

        y_trn_flat = (y_trn_flat - y_min_) / y_max_
        y_vld_flat = (y_vld_flat - y_min_) / y_max_

        y_train = (y_train - y_min_) / y_max_
        y_valid = (y_valid - y_min_) / y_max_

        print("X_train:", X_train.shape)
        print("X_valid:", X_valid.shape)
        print("X_train_flat:", X_trn_flat.shape)
        print("X_valid_flat:", X_vld_flat.shape)

        print()

        print("y_train:", y_train.shape)
        print("y_valid:", y_valid.shape)
        print("y_trn_flat:", y_trn_flat.shape)
        print("y_vld_flat:", y_vld_flat.shape)

        if task == "floc":
            if "xgb" in model_name:
                dtrain = xgb.DMatrix(X_trn_flat, label=y_trn_flat)
                dtest = xgb.DMatrix(X_vld_flat, label=y_vld_flat)
                if model_name == "lognormal_xgb":
                    xgblss = XGBoostLSS(
                        LogNormal(stabilization="None",  
                                  response_fn="exp",      
                                  loss_fn="nll"))
                elif model_name == "weibull_xgb":
                    xgblss = XGBoostLSS(
                        Weibull(stabilization="None",  
                                response_fn="exp",      
                                loss_fn="nll"))
                elif model_name == "expon_xgb":
                    xgblss = XGBoostLSS(
                        Exponential(stabilization="None",  
                                  response_fn="exp",      
                                  loss_fn="nll"))
                elif model_name == "invgauss_xgb":
                    xgblss = XGBoostLSS(
                        InverseGaussian(stabilization="None",  
                                        response_fn="exp",      
                                        loss_fn="nll"))
                else:
                    raise ValueError("Don't know %s" % model_name)
                
                # opt_params = {
                #     "n_rounds":                 50,
                #     "early_stopping_rounds":    100,
                #     "booster":                  "gbtree",
                #     "max_depth":                13,
                #     "min_child_weight":         39,
                #     "colsample_bytree":         0.2545374925231651,
                #     "colsample_bylevel":        0.6909224923784677,
                #     "lambda":                   31.393252465064943,
                #     "alpha":                    0.24167936088332426,
                #     "learning_rate":            0.008237525103357958,
                #     "n_jobs":                   -1,
                # }

                # HPO USING SMAC3
                ###############################################################
                
                # XGBoost Configuration Space Dictionary - Source: NAS-Bench-301 XGBoost surrogate config space (Zela et al., ICLR 2022)
                # https://github.com/automl/nasbench301/blob/master/nasbench301/surrogate_models/configs/model_configs/gradient_boosting/xgb_configspace.json
                param_dict = {
                    # "n_rounds":                 ["int",         {"low": 10,      "high": 100,    "log": False, "default": 30}],
                    "n_rounds":                 ["constant",    100], # fix number of rounds to 500
                    "early_stopping_rounds":    ["constant",    20],
                    "booster":                  ["constant",    "gbtree"],
                    "max_depth":                ["int",         {"low": 1,      "high": 20,     "log": False, "default": 13}],
                    "min_child_weight":         ["int",         {"low": 1,      "high": 100,    "log": True,  "default": 39}],
                    "colsample_bytree":         ["float",       {"low": 0.0,    "high": 1.0,    "log": False, "default": 0.2545374925231651}],
                    "colsample_bylevel":        ["float",       {"low": 0.0,    "high": 1.0,    "log": False, "default": 0.6909224923784677}],
                    "lambda":                   ["float",       {"low": 0.001,  "high": 1000,   "log": True,  "default": 31.393252465064943}],
                    "alpha":                    ["float",       {"low": 0.001,  "high": 1000,   "log": True,  "default": 0.24167936088332426}],
                    "learning_rate":            ["float",       {"low": 0.001,  "high": 0.1,    "log": True,  "default": 0.008237525103357958}],
                    # "bananas_enc"   :         ["categorical", {"choices": [False], "default": False}],
                    # "eta":                    ["float",      {"low": 1e-5,   "high": 1,     "log": True}],
                    # "gamma":                  ["float",      {"low": 1e-8,   "high": 40,    "log": True}],
                    # "subsample":              ["float",      {"low": 0.2,    "high": 1.0,   "log": False}],
                    "n_jobs":                   ["constant",    -1],
                }
                

                # Configuration Space from XGBoostLSS
                # Default values are taken from NAS benchmark configspace and XGBoost documentation: https://xgboost.readthedocs.io/en/stable/parameter.html
                # param_dict = {
                #     # Parameters available in NAS benchmark configspace (use the same default values)
                #     # "n_rounds":         ["constant", 500],
                #     "eta":              ["float", {"low": 1e-5,   "high": 1,     "log": True,  "default": 0.008237525103357958}], # learning_rate
                #     "colsample_bytree": ["float", {"low": 0.2,    "high": 1.0,   "log": False, "default": 0.2545374925231651}],
                #     "min_child_weight": ["float", {"low": 1e-8,   "high": 500,   "log": True,  "default": 39}],
                #     "booster":          ["constant",    "gbtree"],

                #     # Parameters not available in NAS benchmark configspace (use default values from XGBoost documentation)
                #     "gamma":            ["float", {"low": 1e-8,   "high": 40,    "log": True,  "default": 1.0}], # min_split_loss, default is 0 but use 1.0 since 0 is out of range
                #     "subsample":        ["float", {"low": 0.2,    "high": 1.0,   "log": False, "default": 1.0}], # subsample ratio of training instances
                #     "max_depth":        ["int",   {"low": 1,      "high": 10,    "log": False, "default": 6}],

                #     "n_jobs":           ["constant",    10], # Number of parallel threads used to run XGBoost
                # }

                np.random.seed(args.seed)
                torch.manual_seed(args.seed)

                opt_params = xgblss.hyper_opt_smac(
                    param_dict, dtrain,
                    nfold=10,                    # Number of cv-folds.
                    n_trials=100,                 # The number of trials. If this argument is set to None, there is no limitation on the number of trials.
                    seed=args.seed,              # Seed used to generate cv-folds.
                )
                print(opt_params)

                np.random.seed(args.seed)
                torch.manual_seed(args.seed)

                opt_params = opt_params.copy()
                num_boost_round = opt_params["n_rounds"]        
                del opt_params["n_rounds"]
                del opt_params["early_stopping_rounds"]

                # HPO USING OPTUNA
                ###############################################################

                # param_dict = {
                #     # "n_rounds":               ["int",         {"low": 10,      "high": 100,    "log": False, "default": 30}],
                #     # "n_rounds":               ["int",         {"low": 10,      "high": 50,     "log": False, "default": 20}],
                #     # "early_stopping_rounds":  ["constant",    10],
                #     "booster":                  ["constant",    "gbtree"],
                #     "max_depth":                ["int",         {"low": 1,      "high": 20,     "log": False, "default": 13}],
                #     "min_child_weight":         ["int",         {"low": 1,      "high": 100,    "log": True,  "default": 39}],
                #     "colsample_bytree":         ["float",       {"low": 0.0,    "high": 1.0,    "log": False, "default": 0.2545374925231651}],
                #     "colsample_bylevel":        ["float",       {"low": 0.0,    "high": 1.0,    "log": False, "default": 0.6909224923784677}],
                #     "lambda":                   ["float",       {"low": 0.001,  "high": 1000,   "log": True,  "default": 31.393252465064943}],
                #     "alpha":                    ["float",       {"low": 0.001,  "high": 1000,   "log": True,  "default": 0.24167936088332426}],
                #     "learning_rate":            ["float",       {"low": 0.001,  "high": 0.1,    "log": True,  "default": 0.008237525103357958}],
                #     # "bananas_enc"   :         ["categorical", {"choices": [False], "default": False}],
                #     # "eta":                    ["float",      {"low": 1e-5,   "high": 1,     "log": True}],
                #     # "gamma":                  ["float",      {"low": 1e-8,   "high": 40,    "log": True}],
                #     # "subsample":              ["float",      {"low": 0.2,    "high": 1.0,   "log": False}],
                #     "n_jobs":                   ["constant",    1],
                # }
                # np.random.seed(args.seed)
                # torch.manual_seed(args.seed)
                # opt_params = xgblss.hyper_opt(param_dict,
                #             dtrain,
                #             num_boost_round=500,         # Number of boosting iterations.
                #             nfold=10,                    # Number of cv-folds.
                #             early_stopping_rounds=20,    # Number of early-stopping rounds
                #             max_minutes=60*50,              # Time budget in minutes, i.e., stop study after the given number of minutes.
                #             n_trials=100 ,               # The number of trials. If this argument is set to None, there is no limitation on the number of trials.
                #             silence=False,               # Controls the verbosity of the trail, i.e., user can silence the outputs of the trail.
                #             seed=args.seed,                   # Seed used to generate cv-folds.
                #             hp_seed=args.seed                 # Seed for random number generator used in the Bayesian hyperparameter search.
                #         )
                # print(opt_params)

                # np.random.seed(args.seed)
                # torch.manual_seed(args.seed)

                # opt_params = opt_params.copy()
                # num_boost_round = opt_params["opt_rounds"]        
                # del opt_params["opt_rounds"]

                # TRAINING XGBOOSTLSS
                ###############################################################
                
                xgblss.train(params=opt_params, 
                             dtrain=dtrain, 
                             num_boost_round=num_boost_round)

                dtrain = xgb.DMatrix(X_train, label=y_train)
                dtest = xgb.DMatrix(X_valid, label=y_valid)
                
                tra_pred = xgblss.predict(dtrain, pred_type="parameters")
                val_pred = xgblss.predict(dtest, pred_type="parameters")

                print(tra_pred.shape, val_pred.shape)
            
            else:
                raise NotImplementedError("Don't know %s -> %s, %s" %
                                          (args.model, task, model_name))

            # Cut out zeros
            tra_pred = np.array(tra_pred)
            val_pred = np.array(val_pred)

            dump_res(tra_pred=tra_pred, val_pred=val_pred,
                     save_path=save_path, add_info=add_info)

        else:
            raise NotImplementedError("Don't know %s -> %s, %s" %
                                      (args.model, task, model_name))

def dump_res(save_path, tra_pred, val_pred, add_info):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "wb") as fh:
        pickle.dump([tra_pred, val_pred, add_info], fh,
                    protocol=pickle.HIGHEST_PROTOCOL)
    print("Dumped to %s" % save_path)

if __name__ == "__main__":
    main()

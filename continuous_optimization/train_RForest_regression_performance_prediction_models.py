# Written by Ana Kostovska et al. (GECCO 2023 work), shared privately and published with permission.
# Modified by Hoang Dung Dang: hard-coded paths replaced by DATA_DIR / RESULTS_DIR.

import pandas as pd 
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
import joblib
import os
import sys
import ast

# Paths (originally hard-coded to the authors' cluster). Override with environment variables.
DATA_DIR = os.environ.get("DATA_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "data"))
RESULTS_DIR = os.environ.get("RESULTS_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "results"))

def train_predict(algo, dim, runtime):
    random_state = 42
    performance_table_data = []
    print(f'algo: {algo}, dim: {dim}, runtime: {runtime}')
    # data/results locations are configurable via DATA_DIR / RESULTS_DIR
    df_X = pd.read_csv(f"{DATA_DIR}/processed_data/landscape_data/ELA_LHS_5D_1000n_100seed_reduced.csv", index_col=0)
    df_y = pd.read_csv(f"{DATA_DIR}/processed_data/performance_data_regression_log/{algo}_{dim}D_{runtime}.csv", index_col=0)
    df = df_X.join(df_y)
    X = df.drop(['target'], axis = 1)
    X = X.rename(columns=lambda s: s.split("/")[-1])
    y = df.iloc[:,-1]
    # outer cv loop
    groups_outer = [i.split("_")[1] for i in df.index]
    logo_outer = LeaveOneGroupOut()
    r2_scores_train_outer = []
    r2_scores_test_outer = []
    test_preds = []
    test_preds_index = []
    for train_index_outer, test_index_outer in logo_outer.split(X, y, groups_outer):
        X_train_outer, X_test_outer = X.iloc[train_index_outer], X.iloc[test_index_outer]
        y_train_outer, y_test_outer = y[train_index_outer], y[test_index_outer]
        # iterate over all possible hyperparameter configurations
        best_inner_model = None
        best_r2_score_test_inner = -10000
        for n_estimators in [10, 50, 100]:
            for max_features in [1.0, 'sqrt', 'log2']:
                for max_depth in [4,8,15, None]:
                    for min_samples_split in [2, 5, 10]:
                        # inner cv loop
                        groups_inner = [i.split("_")[1] for i in X_train_outer.index]
                        logo_inner = LeaveOneGroupOut()
                        r2_scores_train_inner = []
                        r2_scores_test_inner = []
                        for train_index_inner, test_index_inner in logo_inner.split(X_train_outer, y_train_outer, groups_inner):
                            X_train_inner, X_test_inner = X_train_outer.iloc[train_index_inner], X_train_outer.iloc[test_index_inner]
                            y_train_inner, y_test_inner = y_train_outer[train_index_inner], y_train_outer[test_index_inner]
                            model = RandomForestRegressor(n_estimators=n_estimators, max_features=max_features, max_depth=max_depth, min_samples_split=min_samples_split, n_jobs=-1, random_state=random_state)
                            model.fit(X_train_inner, y_train_inner)
                            # calculate performance metrics on the inner cv 
                            y_train_inner_pred = model.predict(X_train_inner)
                            y_test_inner_pred = model.predict(X_test_inner)
                            r2_score_train_inner = r2_score(y_train_inner, y_train_inner_pred)
                            r2_score_test_inner = r2_score(y_test_inner, y_test_inner_pred)
                            r2_scores_train_inner.append(r2_score_train_inner)
                            r2_scores_test_inner.append(r2_score_test_inner)
                        mean_of_r2_scores_test_inner = np.mean(r2_scores_test_inner)
                        if best_r2_score_test_inner < mean_of_r2_scores_test_inner:
                            best_r2_score_test_inner = mean_of_r2_scores_test_inner
                            best_inner_model = model
        # refit on the whole outer training data
        best_inner_model.fit(X_train_outer, y_train_outer)
        y_train_outer_pred = best_inner_model.predict(X_train_outer)
        y_test_outer_pred = best_inner_model.predict(X_test_outer)
        test_preds.extend(y_test_outer_pred)
        test_preds_index.extend(y_test_outer.index)
        r2_score_train_outer = r2_score(y_train_outer, y_train_outer_pred)
        r2_score_test_outer = r2_score(y_test_outer, y_test_outer_pred)
        r2_scores_train_outer.append(r2_score_train_outer)
        r2_scores_test_outer.append(r2_score_test_outer)
        # save best model in each inner cv fold
        models_folder = f'{RESULTS_DIR}/regression/RForest/models'
        if not os.path.exists(models_folder):
            os.makedirs(models_folder)
        fold = list(X_test_outer.index)[0].split("_")[1]
        joblib.dump(best_inner_model, f'{RESULTS_DIR}/regression/RForest/models/{algo}_{dim}D_{runtime}_test_fold_{fold}.pkl', compress = 1)
    
    # save predictions on the outer test set
    predictions_folder = f'{RESULTS_DIR}/regression/RForest/predictions'
    if not os.path.exists(predictions_folder):
        os.makedirs(predictions_folder)
    test_preds_df = pd.DataFrame(test_preds, columns=["predicted"])
    test_preds_df.index = test_preds_index
    test_preds_df.to_csv(f'{RESULTS_DIR}/regression/RForest/predictions/{algo}_{dim}D_{runtime}.csv')
    performance_table_data.append([algo, dim, runtime, np.mean(r2_scores_train_outer), np.mean(r2_scores_test_outer)])
    print(f'r2_train: {np.mean(r2_scores_train_outer)}, r2_test: {np.mean(r2_scores_test_outer)}')
    performance_df = pd.DataFrame(performance_table_data, columns=['algo', 'dim', 'runtime','train_r2', 'test_r2'])
    performance_df.to_csv(f"{RESULTS_DIR}/regression/RForest/performance_table_summary_{algo}_{dim}D_{runtime}.csv")



if __name__ == "__main__":
    algo = sys.argv[1]
    dim = int(sys.argv[2])
    runtime = int(sys.argv[3])
    train_predict(algo, dim, runtime)

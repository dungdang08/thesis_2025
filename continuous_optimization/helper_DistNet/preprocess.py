# Adapted from DistNet (https://github.com/KEggensperger/DistNet),
# Eggensperger, Lindauer & Hoos, IJCAI 2018. Modified by Hoang Dung Dang (thesis student);
# see ATTRIBUTION.md for the list of changes.

import numpy as np

EPSILON = 1e-4


def replace_zeros(runningtimes, features=None, sat_ls=None):
    """
    Replace all values <= EPSILON in runningtimes with EPSILON.
    """
    if features is None:
        features = [0] * runningtimes.shape[0]
    if sat_ls is None:
        sat_ls = [0] * runningtimes.shape[0]

    new_rt = np.where(runningtimes < EPSILON, EPSILON, runningtimes)
    return new_rt, features, sat_ls


def remove_zeros(runningtimes, features=None, sat_ls=None):
    """
    Remove all instances with more than one value <= EPSILON
    """

    if features is None:
        features = [0] * runningtimes.shape[0]
    if sat_ls is None:
        sat_ls = [0] * runningtimes.shape[0]

    new_rt = list()
    new_ft = list()
    new_sl = list()
    assert runningtimes.shape[0] == len(features) == len(sat_ls)
    for instance, feature, sat in zip(runningtimes, features, sat_ls):
        if not np.any(instance <= EPSILON):
            new_ft.append(feature)
            new_rt.append(instance)
            new_sl.append(sat)
    print("Discarding %d (%d) instances because of ZEROS" % (len(features) - len(new_ft), len(features)))
    return np.array(new_rt), np.array(new_ft), new_sl


def remove_constant_instances(runningtimes, features, sat_ls=None):
    if sat_ls is None:
        sat_ls = [0] * runningtimes.shape[0]

    new_rt = list()
    new_ft = list()
    new_sl = list()
    assert runningtimes.shape[0] == len(features) == len(sat_ls)
    for f, r, s in zip(features, runningtimes, sat_ls):
        if np.std(f) > 0:
            new_rt.append(r)
            new_sl.append(s)
            new_ft.append(f)
    print("Discarding %d (%d) instances because of constant features" %
          (len(features) - len(new_ft), len(features)))
    return np.array(new_rt), np.array(new_ft), new_sl


def det_constant_features(X):
    """
    Return a list with constant features
    :param X:
    :return:
    """
    max_ = X.max(axis=0)
    min_ = X.min(axis=0)
    diff = max_ - min_

    det_idx = np.where(diff <= 10e-10)
    print("Discarding %d (%d) features" % (det_idx[0].shape[0], X.shape[1]))
    return det_idx


def det_transformation(X):
    """
    Return min max scaling
    """
    min_ = np.min(X, axis=0)
    max_ = np.max(X, axis=0) - min_
    return min_, max_


def preprocess_features(tra_X, val_X, scal="meanstd"):
    # Remove constant features and rescale

    # Remove constant features
    del_idx = det_constant_features(tra_X)
    tra_X = np.delete(tra_X, del_idx, axis=1)
    val_X = np.delete(val_X, del_idx, axis=1)

    # Min/Max Scale instance features
    if scal == "minmax":
        min_, max_ = det_transformation(tra_X)
        tra_X = (tra_X - min_) / max_
        val_X = (val_X - min_) / max_
    else:
        mean_ = tra_X.mean(axis=0)
        std_ = tra_X.std(axis=0)
        tra_X = (tra_X - mean_) / std_
        val_X = (val_X - mean_) / std_

    return tra_X, val_X
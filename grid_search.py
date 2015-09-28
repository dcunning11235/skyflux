import numpy as np
import matplotlib.pyplot as plt

from astropy.table import Table
from astropy.table import Column
from astropy.table import vstack
from astropy.table import join

from sklearn.decomposition import FastICA
from sklearn import ensemble
from sklearn import neighbors
from sklearn.pipeline import Pipeline
from sklearn.pipeline import make_pipeline

from sklearn.grid_search import RandomizedSearchCV

from scipy.stats import randint as sp_randint

from operator import itemgetter

import fnmatch
import os
import os.path
import sys
import random
import pickle

import random_forest_spectra
import ICAize

ica_max_iter=750
random_state=1234975
n_iter_search = 1000

def main():
    path = "."
    metadata_path = ".."

    rfr = Pipeline([ ('ica', FastICA(random_state=random_state, max_iter=ica_max_iter)),
                     ('rfr', ensemble.RandomForestRegressor(random_state=random_state, n_jobs=-1))
                ])

    param_grid = {
        "ica__n_components": sp_randint(15, 200),
        "rfr__n_estimators": sp_randint(25, 400),
        "rfr__min_samples_split": sp_randint(1, 10)
        #,
        #"rfr__max_features": [None, "log2", "sqrt"]
    }

    randsearch = RandomizedSearchCV(rfr, param_grid, n_iter = n_iter_search)

    flux_arr, exp_arr, wavelengths = ICAize.load_all_in_dir(path=path, use_con_flux=True, recombine_flux=False)

    obs_metadata = random_forest_spectra.trim_observation_metadata(random_forest_spectra.load_observation_metadata(metadata_path))
    reduced_obs_metadata = obs_metadata[np.in1d(obs_metadata['EXP_ID'], exp_arr)]
    reduced_obs_metadata.sort('EXP_ID')
    sorted_inds = np.argsort(exp_arr)
    reduced_obs_metadata.remove_column('EXP_ID')
    md_len = len(reduced_obs_metadata)
    X_arr = np.array(reduced_obs_metadata).view('f8').reshape((md_len,-1))

    randsearch.fit(flux_arr[sorted_inds], X_arr)

    top_scores = sorted(randsearch.grid_scores_, key=itemgetter(1), reverse=True)[:5]
    for i, score in enumerate(top_scores):
        print "Model with rank:", i
        print "Mean validation score/std:", score.mean_validation_score, np.std(score.cv_validation_scores)
        print "Parameters:", score.parameters
        print ""

if __name__ == '__main__':
    main()

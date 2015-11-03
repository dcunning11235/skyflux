import numpy as np
import matplotlib.pyplot as plt

from astropy.table import Table
from astropy.table import Column
from astropy.table import vstack
from astropy.table import join

#from sklearn import preprocessing as skpp
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LinearRegression as Linear

from sklearn.linear_model import RidgeCV
from sklearn.linear_model import MultiTaskLassoCV as LassoCV
from sklearn.linear_model import MultiTaskElasticNetCV

from sklearn.linear_model import RANSACRegressor as RANSAC

from sklearn.cross_decomposition import PLSRegression as PLS

from sklearn.cross_validation import train_test_split
from sklearn.metrics import mean_squared_error

from scipy.linalg import svd

import fnmatch
import os
import os.path
import sys
import random
import pickle

from ICAize import load_all_in_dir
from random_forest_spectra import trim_observation_metadata, load_observation_metadata

metadata_path = ".."
spectra_path = "."

def main():
    obs_metadata = trim_observation_metadata(load_observation_metadata(metadata_path))
    flux_arr, exp_arr, ivar_arr, mask_arr, wavelengths = load_all_in_dir(spectra_path, recombine_flux=True)

    sorted_inds = np.argsort(exp_arr)

    reduced_obs_metadata = obs_metadata[np.in1d(obs_metadata['EXP_ID'], exp_arr)]
    reduced_obs_metadata.sort('EXP_ID')
    md_len = len(reduced_obs_metadata)

    X_arr = np.array(reduced_obs_metadata).view('f8').reshape((md_len,-1))

    X_arr_train, X_arr_test, flux_arr_train, flux_arr_test = \
        train_test_split(X_arr, flux_arr[sorted_inds], test_size=0.15)


    u, s, vh = svd(ivar_arr, full_matrices=True)

    '''
    linear = Linear(fit_intercept=True, copy_X=True)
    poly_linear = Pipeline([('poly', PolynomialFeatures(degree=2)),
                        ('linear', Linear(fit_intercept=True, copy_X=True))])
    linear.fit(X_arr_train, flux_arr_train, n_jobs=-1)
    poly_linear.fit(X_arr_train, flux_arr_train, linear__n_jobs=-1)

    lin_predictions = linear.predict(X_arr_test)
    mse = mean_squared_error(flux_arr_test, lin_predictions)
    print mse
    plin_predictions = poly_linear.predict(X_arr_test)
    mse = mean_squared_error(flux_arr_test, plin_predictions)
    print mse

    ridge = RidgeCV()
    poly_ridge = Pipeline([('poly', PolynomialFeatures(degree=2)),
                        ('ridge', RidgeCV())])
    ridge.fit(X_arr_train, flux_arr_train)
    poly_ridge.fit(X_arr_train, flux_arr_train)

    ridge_predictions = ridge.predict(X_arr_test)
    mse = mean_squared_error(flux_arr_test, ridge_predictions)
    print mse
    pridge_predictions = poly_ridge.predict(X_arr_test)
    mse = mean_squared_error(flux_arr_test, pridge_predictions)
    print mse
    '''

    lasso = LassoCV(n_jobs=-1)
    poly_lasso = Pipeline([('poly', PolynomialFeatures(degree=2)),
                        ('lasso', LassoCV(n_jobs=-1))])
    lasso.fit(X_arr_train, flux_arr_train)
    poly_lasso.fit(X_arr_train, flux_arr_train)

    lasso_predictions = lasso.predict(X_arr_test)
    mse = mean_squared_error(flux_arr_test, lasso_predictions)
    print mse
    plasso_predictions = poly_lasso.predict(X_arr_test)
    mse = mean_squared_error(flux_arr_test, plasso_predictions)
    print mse

    elastic = ElasticNetCV(n_jobs=-1)
    poly_elastic = Pipeline([('poly', PolynomialFeatures(degree=2)),
                        ('elastic', ElasticNetCV(n_jobs=-1))])
    elastic.fit(X_arr_train, flux_arr_train)
    poly_elastic.fit(X_arr_train, flux_arr_train)

    elastic_predictions = elastic.predict(X_arr_test)
    mse = mean_squared_error(flux_arr_test, elastic_predictions)
    print mse
    pelastic_predictions = poly_elastic.predict(X_arr_test)
    mse = mean_squared_error(flux_arr_test, pelastic_predictions)
    print mse

    pls = PLS(n_components=8)
    pls_predictions = pls.predict(X_arr_test)
    mse = mean_squared_error(flux_arr_test, pls_predictions)

    '''
    plt.legend(['Actual', 'Predicted', 'Delta'])
    plt.plot(wavelengths, flux_arr_test[-1])
    plt.plot(wavelengths, plin_predictions[-1])
    plt.plot(wavelengths, plin_predictions[-1]-flux_arr_test[-1])
    plt.tight_layout()
    plt.show()
    plt.close()

    plt.legend(['Actual', 'Predicted', 'Delta'])
    plt.plot(wavelengths, flux_arr_test[-1])
    plt.plot(wavelengths, pridge_predictions[-1])
    plt.plot(wavelengths, pridge_predictions[-1]-flux_arr_test[-1])
    plt.tight_layout()
    plt.show()
    plt.close()
    '''


if __name__ == '__main__':
    main()

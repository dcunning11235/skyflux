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

#from sklearn.cross_decomposition import PLSRegression as PLS

#from sklearn.gaussian_process import GaussianProcess as GaussianProcess

from sklearn.cross_validation import train_test_split
#from sklearn.cross_validation import PredefinedSplit
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

try_weighted = False
subtract_mean = False #True

restrict_delta = True

def main():
    obs_metadata = trim_observation_metadata(load_observation_metadata(metadata_path))
    flux_arr, exp_arr, ivar_arr, mask_arr, wavelengths = load_all_in_dir(spectra_path, recombine_flux=True)

    sorted_inds = np.argsort(exp_arr)

    reduced_obs_metadata = obs_metadata[np.in1d(obs_metadata['EXP_ID'], exp_arr)]
    reduced_obs_metadata.sort('EXP_ID')
    md_len = len(reduced_obs_metadata)

    X_arr = np.array(reduced_obs_metadata).view('f8').reshape((md_len,-1))

    '''
    test_fold = np.zeros((X_arr.shape[0], ), dtype=int)
    test_fold[600:]=1
    ps = PredefinedSplit(test_fold=test_fold)
    print len(ps)
    '''

    test_inds = range(0, 600)
    linear = Linear(fit_intercept=True, copy_X=True, n_jobs=-1)
    poly_linear = Pipeline([('poly', PolynomialFeatures(degree=2)),
                        ('linear', Linear(fit_intercept=True, copy_X=True, n_jobs=-1))])


    for test_ind in test_inds:
        test_X = X_arr[test_ind]
        train_X = np.vstack( [X_arr[:test_ind], X_arr[test_ind+1:]] )
        test_y =  (flux_arr[sorted_inds])[test_ind]
        train_y = np.vstack( [(flux_arr[sorted_inds])[:test_ind], (flux_arr[sorted_inds])[test_ind+1:]] )

        linear.fit(train_X, train_y)
        poly_linear.fit(train_X, train_y)

        lin_predictions = linear.predict(test_X)[0]
        plin_predictions = poly_linear.predict(test_X)[0]

        mask = (ivar_arr[test_ind] == 0) | np.isclose(lin_predictions, 0)
        if restrict_delta:
            delta_mask = mask.copy()
            delta_mask[:2700] = True
        else:
            delta_mask = mask

        lin_delta = lin_predictions - test_y
        err_term = np.sum(np.power(lin_delta[~delta_mask], 2))/len(wavelengths[~delta_mask])
        err_sum = np.sum(lin_delta[~delta_mask])/len(lin_delta[~delta_mask])
        print err_term, err_sum,

        plin_delta = plin_predictions - test_y
        err_term = np.sum(np.power(plin_delta[~delta_mask], 2))/len(wavelengths[~delta_mask])
        err_sum = np.sum(plin_delta[~delta_mask])/len(plin_delta[~delta_mask])
        print err_term, err_sum



    '''
    ransac = RANSAC()
    poly_ransac = Pipeline([('poly', PolynomialFeatures(degree=2)),
                        ('ransac', RANSAC())])
    print X_arr_train.shape, flux_arr_train.shape
    ransac.fit(np.copy(X_arr_train), np.copy(flux_arr_train))
    poly_ransac.fit(X_arr_train, flux_arr_train)

    r_predictions = ransac.predict(X_arr_test)
    mse = mean_squared_error(flux_arr_test, r_predictions)
    print mse
    pr_predictions = poly_ransac.predict(X_arr_test)
    mse = mean_squared_error(flux_arr_test, pr_predictions)
    print mse
    '''

    '''
    #gp = GaussianProcess(nugget=np.power(flux_arr_train, 2)/ivar_train) #regr="quadratic")
    gp = GaussianProcess()
    gp.fit(X_arr_train, flux_arr_train)
    gp_predictions = gp.predict(X_arr_test)
    mse = mean_squared_error(flux_arr_test, gp_predictions)
    print mse
    '''

    '''
    del lin_predictions
    del plin_predictions
    del linear
    del poly_linear

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

    del ridge_predictions
    del pridge_predictions
    del ridge
    del poly_ridge

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

    del lasso_predictions
    del plasso_predictions
    del lasso
    del poly_lasso

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

    del elastic_predictions
    del pelastic_predictions
    del elastic
    del poly_elastic
    '''

    '''
    pls = PLS(n_components=8, max_iter=2000)
    pls.fit(X_arr_train, flux_arr_train)
    pls_predictions = pls.predict(X_arr_test)
    mse = mean_squared_error(flux_arr_test, pls_predictions)
    '''
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

import numpy as np
import matplotlib.pyplot as plt

from astropy.table import Table
from astropy.table import Column
from astropy.table import vstack
from astropy.table import join

from sklearn import ensemble
from sklearn import linear_model
from sklearn import neighbors

from sklearn.decomposition import FastICA
from sklearn.decomposition import SparsePCA
from sklearn import preprocessing as skpp
from sklearn.pipeline import make_pipeline
from sklearn.cross_decomposition import PLSRegression

from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LinearRegression as Linear

import ICAize

import fnmatch
import os
import os.path
import sys

rfr_random_state = 456371
hide_plots=True #False
use_spca=False #True

include_linear = True
linear_only = True
order_3 = False
order_4 = False

def load_all_spectra_data(path, noncon_file=None, con_file=None, use_spca=False):
    nc_sources, nc_mixing, noncon_exposures = load_spectra_data(path,
                                target_type='noncontinuum', filename=noncon_file,
				use_spca=use_spca)
    c_sources, c_mixing, c_exposures = load_spectra_data(path,
                                target_type='continuum', filename=con_file,
				use_spca=use_spca)

    return c_sources, c_mixing, c_exposures, nc_sources, nc_mixing, noncon_exposures

def load_spectra_data(path, target_type='continuum', filename=None, use_spca=False):
    if filename is None:
        if use_spca:
            filename = ICAize.spca_data_file.format(target_type)
        else:
            filename = ICAize.ica_data_file.format(target_type)
    npz = np.load(os.path.join(path, filename))

    sources = npz['sources']
    if not use_spca:
        mixing = npz['mixing']
    else:
        mixing = None
    exposures = npz['exposures']
    wavelengths = npz['wavelengths']

    npz.close()
    return sources, mixing, exposures, wavelengths

def load_observation_metadata(path, file = "annotated_metadata.csv"):
    data = Table.read(os.path.join(path, file), format="ascii.csv")

    return data

def trim_observation_metadata(data, copy=False):
    if copy:
        data = data.copy()

    kept_columns = ['EXP_ID', 'RA', 'DEC', 'AZ', 'ALT', 'AIRMASS', #'TIME_BLOCK','WEEK_OF_YEAR',
                    'LUNAR_MAGNITUDE', 'LUNAR_ELV', 'LUNAR_SEP', 'SOLAR_ELV',
                    'SOLAR_SEP', 'GALACTIC_CORE_SEP', 'GALACTIC_PLANE_SEP', 'SS_COUNT', 'SS_AREA']
    removed_columns = [name for name in data.colnames if name not in kept_columns]
    data.remove_columns(removed_columns)

    return data

def main():
    metadata_path = ".."
    spectra_path = "."
    test_ind = 0
    target_types = 'combined'
    #target_types = 'continuum'

    if len(sys.argv) == 2:
        metadata_path = sys.argv[1]
    if len(sys.argv) == 3:
        metadata_path = sys.argv[1]
        spectra_path = sys.argv[2]
    if len(sys.argv) == 4:
        metadata_path = sys.argv[1]
        spectra_path = sys.argv[2]
        test_inds = sys.argv[3].split(",")
    if len(sys.argv) == 5:
        metadata_path = sys.argv[1]
        spectra_path = sys.argv[2]
        test_inds = sys.argv[3].split(",")
        target_types = sys.argv[4]

    if target_types == 'both':
        target_types = ['continuum', 'noncontinuum']
    else:
        target_types = [target_types]

    wavelengths = None
    for target_type in target_types:
        if test_inds[0] == 'ALL':
            test_inds = range(int(test_inds[1]), int(test_inds[2]))
        else:
            test_inds = [int(test_inds[0])]

        load_plot_etc_target_type(metadata_path, spectra_path, test_inds, target_type,
					no_plot=hide_plots, save_out=True, restrict_delta=True,
					use_spca=use_spca)

def load_plot_etc_target_type(metadata_path, spectra_path, test_inds, target_type, no_plot=False,
				save_out=False, restrict_delta=False, use_spca=False):
    obs_metadata = trim_observation_metadata(load_observation_metadata(metadata_path))
    c_sources, c_mixing, c_exposures, c_wavelengths = load_spectra_data(spectra_path,
						target_type=target_type, use_spca=use_spca)

    reduced_obs_metadata = obs_metadata[np.in1d(obs_metadata['EXP_ID'], c_exposures)]
    reduced_obs_metadata.sort('EXP_ID')
    sorted_inds = np.argsort(c_exposures)

    errs = np.zeros((7,), dtype=float)

    if not linear_only:
        rfr = ensemble.RandomForestRegressor(n_estimators=300, min_samples_split=1,
                        random_state=rfr_random_state, n_jobs=-1, verbose=False)
        knn = neighbors.KNeighborsRegressor(weights='distance', n_neighbors=10, p=64)

    if include_linear:
        linear = Linear(fit_intercept=True, copy_X=True, n_jobs=-1)
        poly_2_linear = Pipeline([('poly', PolynomialFeatures(degree=2)),
                            ('linear', Linear(fit_intercept=True, copy_X=True, n_jobs=-1))])
        poly_3_linear = Pipeline([('poly', PolynomialFeatures(degree=3)),
                        ('linear', Linear(fit_intercept=True, copy_X=True, n_jobs=-1))])
        poly_4_linear = Pipeline([('poly', PolynomialFeatures(degree=4)),
                        ('linear', Linear(fit_intercept=True, copy_X=True, n_jobs=-1))])

    reduced_obs_metadata.remove_column('EXP_ID')
    #reduced_obs_metadata.remove_column('AIRMASS')
    md_len = len(reduced_obs_metadata)
    X_arr = np.array(reduced_obs_metadata).view('f8').reshape((md_len,-1))

    ica = None
    if not use_spca:
        ica = ICAize.unpickle_FastICA(path=spectra_path, target_type=target_type)
    else:
        ica = ICAize.unpickle_SPCA(path=spectra_path, target_type=target_type)

    ################################################################
    results = None
    for test_ind in test_inds:
        test_X = X_arr[test_ind]
        train_X = np.vstack( [X_arr[:test_ind], X_arr[test_ind+1:]] )
        test_y =  (c_sources[sorted_inds])[test_ind]
        train_y = np.vstack( [(c_sources[sorted_inds])[:test_ind], (c_sources[sorted_inds])[test_ind+1:]] )

        title_str = "exp{}, {}".format(c_exposures[sorted_inds[test_ind]], target_type)

        if not linear_only:
            rfr.fit(X=train_X, y=train_y)
            knn.fit(X=train_X, y=train_y)

        if include_linear:
            linear.fit(train_X, train_y)
            poly_2_linear.fit(train_X, train_y)
            if order_3:
                poly_3_linear.fit(train_X, train_y)
            if order_4:
                poly_4_linear.fit(train_X, train_y)

        print test_ind, c_exposures[sorted_inds[test_ind]],

        data = None
        actual = None
        mask = None
        delta_mask = None

        for file in os.listdir(spectra_path):
            #if fnmatch.fnmatch(file, "stacked_sky_*exp{}-continuum.csv".format(c_exposures[sorted_inds[test_ind]])):
            if fnmatch.fnmatch(file, "stacked_sky_*exp{}.csv".format(c_exposures[sorted_inds[test_ind]])):
                data = Table.read(os.path.join(spectra_path, file), format="ascii.csv")
                mask = (data['ivar'] == 0)
                if restrict_delta:
                    delta_mask = mask.copy()
                    delta_mask[:2700] = True
                else:
                    delta_mask = mask

                actual = data['flux']
                '''
                if target_type == 'continuum':
                    actual = data['con_flux']
                elif target_type == 'combined':
                    actual += data['con_flux']
                '''
                break
        if actual is None:
            continue

        if not linear_only:
            rfr_prediction = rfr.predict(test_X)
            if not use_spca:
                rfr_predicted = ica.inverse_transform(rfr_prediction, copy=True)
            else:
                rfr_predicted = np.zeros( (1, ica.components_.shape[1]) )
                rfr_predicted[0,:] = np.sum(rfr_prediction.T * ica.components_, 0)

            rfr_delta = rfr_predicted[0] - actual
            if not no_plot:
                plt.plot(c_wavelengths[~mask], rfr_predicted[0][~mask])
                plt.plot(c_wavelengths[~mask], actual[~mask])
                plt.plot(c_wavelengths[~mask], rfr_delta[~mask])
            if not no_plot:
                plt.plot(c_wavelengths, [0]*len(c_wavelengths))
            err_term = np.sum(np.power(rfr_delta[~delta_mask], 2))/len(c_wavelengths[~delta_mask])
            err_sum = np.sum(rfr_delta[~delta_mask])/len(rfr_delta[~delta_mask])
            if not no_plot:
                plt.legend(['Predicted', 'Actual', 'Delta {:0.5f}'.format(err_term)])
                plt.tight_layout()
                plt.title("Random Forest Regressor: {}".format(title_str))
                plt.show()
                plt.close()
            print err_term, err_sum,
            errs[0] = err_term

            knn_prediction = knn.predict(test_X)
            if not use_spca:
                knn_predicted = ica.inverse_transform(knn_prediction, copy=True)
            else:
                knn_predicted = np.zeros( (1, ica.components_.shape[1]) )
                knn_predicted[0,:] = np.sum(knn_prediction.T * ica.components_, 0)

            if not no_plot:
                plt.plot(c_wavelengths[~mask], knn_predicted[0][~mask])
                plt.plot(c_wavelengths[~mask], actual[~mask])
            knn_delta = knn_predicted[0] - actual
            err_term = np.sum(np.power(knn_delta[~delta_mask], 2))/len(c_wavelengths[~delta_mask])
            err_sum = np.sum(knn_delta[~delta_mask])/len(knn_delta[~delta_mask])
            if not no_plot:
                plt.plot(c_wavelengths[~mask], knn_delta[~mask])
                plt.plot(c_wavelengths, [0]*len(c_wavelengths))
                plt.legend(['Predicted', 'Actual', 'Delta {:0.5f}'.format(err_term)])
                plt.tight_layout()
                plt.title("Good 'ol K-NN: {}".format(title_str))
                plt.show()
                plt.close()
            print err_term, err_sum,
            errs[1] = err_term

            avg_predicted = (knn_predicted + rfr_predicted)/2

            if not no_plot:
                plt.plot(c_wavelengths[~mask], avg_predicted[0][~mask])
                plt.plot(c_wavelengths[~mask], actual[~mask])
            avg_delta = avg_predicted[0] - actual
            err_term = np.sum(np.power(avg_delta[~delta_mask], 2))/len(c_wavelengths[~delta_mask])
            err_sum = np.sum(avg_delta[~delta_mask])/len(avg_delta[~delta_mask])
            if not no_plot:
                plt.plot(c_wavelengths[~mask], avg_delta[~mask])
                plt.plot(c_wavelengths, [0]*len(c_wavelengths))
                plt.legend(['Predicted', 'Actual', 'Delta {:0.5f}'.format(err_term)])
                plt.tight_layout()
                plt.title("Avg. of KNN,RFR: {}".format(title_str))
                plt.show()
                plt.close()
            if include_linear:
                print err_term, err_sum,
            else:
                print err_term, err_sum
            errs[2] = err_term

        if include_linear:
            poly_1_prediction = linear.predict(test_X)
            if not use_spca:
                poly_1_predicted = ica.inverse_transform(poly_1_prediction, copy=True)
            else:
                poly_1_predicted = np.zeros( (1, ica.components_.shape[1]) )
                poly_1_predicted[0,:] = np.sum(poly_1_prediction.T * ica.components_, 0)

            poly_1_delta = poly_1_predicted[0] - actual
            err_term = np.sum(np.power(poly_1_delta[~delta_mask], 2))/len(c_wavelengths[~delta_mask])
            err_sum = np.sum(poly_1_delta[~delta_mask])/len(poly_1_delta[~delta_mask])
            print err_term, err_sum,
            errs[3] = err_term


            poly_2_prediction = poly_2_linear.predict(test_X)
            if not use_spca:
                poly_2_predicted = ica.inverse_transform(poly_2_prediction, copy=True)
            else:
                poly_2_predicted = np.zeros( (1, ica.components_.shape[1]) )
                poly_2_predicted[0,:] = np.sum(poly_2_prediction.T * ica.components_, 0)

            poly_2_delta = poly_2_predicted[0] - actual
            err_term = np.sum(np.power(poly_2_delta[~delta_mask], 2))/len(c_wavelengths[~delta_mask])
            err_sum = np.sum(poly_2_delta[~delta_mask])/len(poly_2_delta[~delta_mask])
            print err_term, err_sum,
            errs[4] = err_term

            if order_3:
                poly_3_prediction = poly_3_linear.predict(test_X)
                if not use_spca:
                    poly_3_predicted = ica.inverse_transform(poly_3_prediction, copy=True)
                else:
                    poly_3_predicted = np.zeros( (1, ica.components_.shape[1]) )
                    poly_3_predicted[0,:] = np.sum(poly_3_prediction.T * ica.components_, 0)

                poly_3_delta = poly_3_predicted[0] - actual
                err_term = np.sum(np.power(poly_3_delta[~delta_mask], 2))/len(c_wavelengths[~delta_mask])
                err_sum = np.sum(poly_3_delta[~delta_mask])/len(poly_3_delta[~delta_mask])
                print err_term, err_sum,
                errs[5] = err_term

            if order_4:
                poly_4_prediction = poly_4_linear.predict(test_X)
                if not use_spca:
                    poly_4_predicted = ica.inverse_transform(poly_4_prediction, copy=True)
                else:
                    poly_4_predicted = np.zeros( (1, ica.components_.shape[1]) )
                    poly_4_predicted[0,:] = np.sum(poly_4_prediction.T * ica.components_, 0)

                poly_4_delta = poly_4_predicted[0] - actual
                err_term = np.sum(np.power(poly_4_delta[~delta_mask], 2))/len(c_wavelengths[~delta_mask])
                err_sum = np.sum(poly_4_delta[~delta_mask])/len(poly_4_delta[~delta_mask])
                print err_term, err_sum,
                errs[6] = err_term

        if save_out:
            out_table = Table()
            wavelength_col = Column(c_wavelengths, name="wavelength", dtype=float)
            out_table.add_columns([wavelength_col])

            if not linear_only:
                rf_col = Column(rfr_predicted[0], name="rf_flux", dtype=float)
                knn_col = Column(knn_predicted[0], name="knn_flux", dtype=float)
                avg_col = Column(avg_predicted[0], name="avg_flux", dtype=float)
                out_table.add_columns([rf_col, knn_col, avg_col])

            if include_linear:
                poly_1_col = Column(poly_1_predicted[0], name="poly_1_flux", dtype=float)
                poly_2_col = Column(poly_2_predicted[0], name="poly_2_flux", dtype=float)
                out_table.add_columns([poly_1_col, poly_2_col])
                if order_3:
                    poly_3_col = Column(poly_3_predicted[0], name="poly_3_flux", dtype=float)
                    out_table.add_columns([poly_3_col])
                if order_4:
                    poly_4_col = Column(poly_4_predicted[0], name="poly_4_flux", dtype=float)
                    out_table.add_columns([poly_4_col])

            mask_col = Column(~mask, name="mask_col", dtype=bool)
            out_table.add_columns([mask_col])

            out_table.write("predicted_sky_exp{}.csv".format(c_exposures[sorted_inds[test_ind]]), format="ascii.csv")


if __name__ == '__main__':
    main()

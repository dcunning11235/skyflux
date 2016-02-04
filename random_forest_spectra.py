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
from sklearn.preprocessing import StandardScaler
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
use_pca=False #True
restrict_delta = True
restrict_color = 'blue'
include_knn = False #True
include_linear = False #True
linear_only = False #True
order_3 = False #True
order_4 = False #True #False
n_estimators = 200
save_out = False
scale = False #True
scale_std = False
proc_ecliptic_angles = False
proc_lunar_sep = False #True
proc_lunar_mag = False #True
reg_type = 'etr'
bootstrap = False
min_samples_split = 1
compute_chi_fit = True
use_filter_split = False

def load_all_spectra_data(path, noncon_file=None, con_file=None, use_spca=False, use_pca=False):
    nc_sources, nc_mixing, noncon_exposures = load_spectra_data(path,
                                target_type='noncontinuum', filename=noncon_file,
				use_spca=use_spca)
    c_sources, c_mixing, c_exposures = load_spectra_data(path,
                                target_type='continuum', filename=con_file,
				use_spca=use_spca)

    return c_sources, c_mixing, c_exposures, nc_sources, nc_mixing, noncon_exposures

def load_spectra_data(path, target_type='continuum', filter_str='em', filename=None, use_spca=False, use_pca=False):
    if filename is None:
        if use_spca:
            filename = ICAize.spca_data_file.format(target_type, filter_str)
        elif use_pca:
            filename = ICAize.pca_data_file.format(target_type, filter_str)
        else:
            filename = ICAize.ica_data_file.format(target_type, filter_str)
    npz = np.load(os.path.join(path, filename))

    sources = npz['sources']
    if not use_spca and not use_pca:
        mixing = npz['mixing']
    else:
        mixing = None
    exposures = npz['exposures']
    wavelengths = npz['wavelengths']
    filter_split_arr = npz['filter_split_arr']

    npz.close()
    return sources, mixing, exposures, wavelengths, filter_split_arr

def load_observation_metadata(path, file = "annotated_metadata.csv"):
    data = Table.read(os.path.join(path, file), format="ascii.csv")

    if proc_ecliptic_angles:
        data['ECLIPTIC_PLANE_SOLAR_SEP'] = np.sqrt(np.pi) - np.sqrt(data['ECLIPTIC_PLANE_SOLAR_SEP'] / 180 * np.pi) + np.sin(data['ECLIPTIC_PLANE_SOLAR_SEP'] / 360 * np.pi)
        data['ECLIPTIC_PLANE_SEP'] = np.expm1(np.sqrt(np.pi) - np.sqrt(np.abs(data['ECLIPTIC_PLANE_SEP']) / 180 * np.pi) ) / np.exp(np.sqrt(np.pi))

    if proc_lunar_mag:
        data['LUNAR_MAGNITUDE'] = np.power(2.512, -data['LUNAR_MAGNITUDE'])

    if proc_lunar_sep:
        #data['LUNAR_SEP'] = 5405*np.power(data['LUNAR_SEP'],-1.25) - 0.00019*np.power(data['LUNAR_SEP'],3) + 0.07085*np.power(data['LUNAR_SEP'],2) - 8.5*data['LUNAR_SEP'] + 340
        data['LUNAR_SEP'] = np.cos(data['LUNAR_SEP']) * np.abs(data['LUNAR_MAGNITUDE']) * (data['LUNAR_ELV'] > 0)

    return data

def trim_observation_metadata(data, copy=False):
    if copy:
        data = data.copy()

    kept_columns = ['EXP_ID', 'RA', 'DEC',
                    #'AZ_COR',
                    'AZ',
                    'ALT', 'AIRMASS', #'TIME_BLOCK','WEEK_OF_YEAR',
                    'LUNAR_MAGNITUDE', 'LUNAR_ELV', 'LUNAR_SEP', 'SOLAR_ELV',
                    'SOLAR_SEP', 'GALACTIC_CORE_SEP',
                    #'GALACTIC_PLANE_SEP_COR',
                    'GALACTIC_PLANE_SEP',
                    'SS_COUNT', 'SS_AREA',
                    #'ECLIPTIC_PLANE_SEP_COR',
                    'ECLIPTIC_PLANE_SEP',
                    'ECLIPTIC_PLANE_SOLAR_SEP'] #,
                    #'BLOCKS_TO_TWILIGHT']
    removed_columns = [name for name in data.colnames if name not in kept_columns]
    data.remove_columns(removed_columns)

    return data

def main():
    metadata_path = ".."
    spectra_path = "."
    test_inds = []
    target_types = 'combined'
    #target_types = 'continuum'

    if len(sys.argv) == 2:
        #metadata_path = sys.argv[1]
        test_inds = [sys.argv[1]]
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
					no_plot=hide_plots, save_out=save_out, restrict_delta=restrict_delta,
					use_spca=use_spca, use_pca=use_pca)

def load_plot_etc_target_type(metadata_path, spectra_path, test_inds, target_type, no_plot=False,
				save_out=False, restrict_delta=False, use_spca=False, use_pca=False):
    obs_metadata = trim_observation_metadata(load_observation_metadata(metadata_path))
    if use_filter_split:
        c_sources, c_mixing, c_exposures, c_wavelengths, c_filter_split_arr = load_spectra_data(spectra_path,
						target_type=target_type, filter_str='nonem', use_spca=use_spca, use_pca=use_pca)
        c_sources_e, c_mixing_e, c_exposures_e, c_wavelengths_e, c_filter_split_arr_e = load_spectra_data(spectra_path,
						target_type=target_type, filter_str='em', use_spca=use_spca, use_pca=use_pca)
    else:
        c_sources, c_mixing, c_exposures, c_wavelengths, c_filter_split_arr = load_spectra_data(spectra_path,
						target_type=target_type, filter_str='both', use_spca=use_spca, use_pca=use_pca)

    reduced_obs_metadata = obs_metadata[np.in1d(obs_metadata['EXP_ID'], c_exposures)]
    reduced_obs_metadata.sort('EXP_ID')
    sorted_inds = np.argsort(c_exposures)
    if use_filter_split:
        sorted_e_inds = np.argsort(c_exposures_e)

    if not linear_only:
        if reg_type == 'etr':
            rfr = ensemble.ExtraTreesRegressor(n_estimators=n_estimators, min_samples_split=min_samples_split,
                        random_state=rfr_random_state, n_jobs=-1, verbose=False, bootstrap=bootstrap)
            if use_filter_split:
                rfr_e = ensemble.ExtraTreesRegressor(n_estimators=n_estimators, min_samples_split=min_samples_split,
                        random_state=rfr_random_state, n_jobs=-1, verbose=False, bootstrap=bootstrap)
        else:
            rfr = ensemble.RandomForestRegressor(n_estimators=n_estimators, min_samples_split=min_samples_split,
                        random_state=rfr_random_state, n_jobs=-1, verbose=False, bootstrap=bootstrap)
            if use_filter_split:
                rfr_e = ensemble.RandomForestRegressor(n_estimators=n_estimators, min_samples_split=min_samples_split,
                        random_state=rfr_random_state, n_jobs=-1, verbose=False, bootstrap=bootstrap)
        if include_knn:
            knn = neighbors.KNeighborsRegressor(weights='distance', n_neighbors=10, p=64)
            if use_filter_split:
                knn_e = neighbors.KNeighborsRegressor(weights='distance', n_neighbors=10, p=64)

    if include_linear:
        linear = Linear(fit_intercept=True, copy_X=True, n_jobs=-1)
        poly_2_linear = Pipeline([('poly', PolynomialFeatures(degree=2)),
                            ('linear', Linear(fit_intercept=True, copy_X=True, n_jobs=-1))])
        poly_3_linear = Pipeline([('poly', PolynomialFeatures(degree=3)),
                        ('linear', Linear(fit_intercept=True, copy_X=True, n_jobs=-1))])
        poly_4_linear = Pipeline([('poly', PolynomialFeatures(degree=4)),
                        ('linear', Linear(fit_intercept=True, copy_X=True, n_jobs=-1))])
        if use_filter_split:
            linear_e = Linear(fit_intercept=True, copy_X=True, n_jobs=-1)
            poly_2_linear_e = Pipeline([('poly', PolynomialFeatures(degree=2)),
                            ('linear', Linear(fit_intercept=True, copy_X=True, n_jobs=-1))])
            poly_3_linear_e = Pipeline([('poly', PolynomialFeatures(degree=3)),
                        ('linear', Linear(fit_intercept=True, copy_X=True, n_jobs=-1))])
            poly_4_linear_e = Pipeline([('poly', PolynomialFeatures(degree=4)),
                        ('linear', Linear(fit_intercept=True, copy_X=True, n_jobs=-1))])

    reduced_obs_metadata.remove_column('EXP_ID')
    md_len = len(reduced_obs_metadata)
    var_count = len(reduced_obs_metadata.columns)
    X_arr = np.array(reduced_obs_metadata).view('f8').reshape((md_len,-1))

    ica = None
    if not use_spca and not use_pca:
        if use_filter_split:
            ica = ICAize.unpickle_FastICA(path=spectra_path, target_type=target_type, filter_str='nonem')
            ica_e = ICAize.unpickle_FastICA(path=spectra_path, target_type=target_type, filter_str='em')
        else:
            ica = ICAize.unpickle_FastICA(path=spectra_path, target_type=target_type, filter_str='both')
    elif use_spca:
        ica = ICAize.unpickle_SPCA(path=spectra_path, target_type=target_type)
    else:
        if use_filter_split:
            ica = ICAize.unpickle_PCA(path=spectra_path, target_type=target_type, filter_str='nonem')
            ica_e = ICAize.unpickle_PCA(path=spectra_path, target_type=target_type, filter_str='em')
        else:
            ica = ICAize.unpickle_PCA(path=spectra_path, target_type=target_type, filter_str='both')

    spectra_dir_list = os.listdir(spectra_path)

    ################################################################
    results = None
    for test_ind in test_inds:
        test_X = X_arr[test_ind]
        train_X = np.vstack( [X_arr[:test_ind], X_arr[test_ind+1:]] )
        test_y =  (c_sources[sorted_inds])[test_ind]
        train_y = np.vstack( [(c_sources[sorted_inds])[:test_ind], (c_sources[sorted_inds])[test_ind+1:]] )
        if use_filter_split:
            test_y_e =  (c_sources_e[sorted_e_inds])[test_ind]
            train_y_e = np.vstack( [(c_sources_e[sorted_e_inds])[:test_ind], (c_sources_e[sorted_e_inds])[test_ind+1:]] )

        if scale:
            scaler = StandardScaler(with_std=scale_std)
            train_X = scaler.fit_transform(train_X)
            test_X = scaler.transform(test_X)

        title_str = "exp{}, {}".format(c_exposures[sorted_inds[test_ind]], target_type)

        if not linear_only:
            rfr.fit(X=train_X, y=train_y)
            if use_filter_split:
                rfr_e.fit(X=train_X, y=train_y_e)
            if include_knn:
                knn.fit(X=train_X, y=train_y)
                if user_filter_split:
                    knn_e.fit(X=train_X, y=train_y_e)

        if include_linear:
            linear.fit(train_X, train_y)
            poly_2_linear.fit(train_X, train_y)
            if order_3:
                poly_3_linear.fit(train_X, train_y)
            if order_4:
                poly_4_linear.fit(train_X, train_y)
        if use_filter_split and include_linear:
            linear_e.fit(train_X, train_y_e)
            poly_2_linear_e.fit(train_X, train_y_e)
            if order_3:
                poly_3_linear_e.fit(train_X, train_y_e)
            if order_4:
                poly_4_linear_e.fit(train_X, train_y_e)

        print test_ind, c_exposures[sorted_inds[test_ind]],

        data = None
        actual = None
        mask = None
        delta_mask = None
        ivar = None

        for file in spectra_dir_list:
            if fnmatch.fnmatch(file, "stacked_sky_*exp{}.csv".format(c_exposures[sorted_inds[test_ind]])):
                data = Table.read(os.path.join(spectra_path, file), format="ascii.csv")
                ivar = data['ivar']
                mask = (data['ivar'] == 0)
                delta_mask = mask.copy()
                if restrict_delta:
                    if restrict_color == 'blue':
                        delta_mask[2700:] = True
                    else:
                        delta_mask[:2700] = True

                actual = data['flux']
                break
        if actual is None:
            continue

        if not linear_only:
            rfr_prediction = rfr.predict(test_X)
            if not use_spca and not use_pca:
                rfr_predicted = ica.inverse_transform(rfr_prediction, copy=True)
            else:
                rfr_predicted = np.zeros( (1, ica.components_.shape[1]) )
                rfr_predicted[0,:] = np.sum(rfr_prediction.T * ica.components_, 0)

            if use_filter_split:
                rfr_e_prediction = rfr_e.predict(test_X)
                if not use_spca and not use_pca:
                    rfr_e_predicted = ica_e.inverse_transform(rfr_e_prediction, copy=True)
                else:
                    rfr_e_predicted = np.zeros( (1, ica_e.components_.shape[1]) )
                    rfr_e_predicted[0,:] = np.sum(rfr_e_prediction.T * ica_e.components_, 0)
                rfr_predicted = rfr_predicted + rfr_e_predicted

            rfr_delta = rfr_predicted[0] - actual
            if not no_plot:
                plt.plot(c_wavelengths[~mask], rfr_predicted[0][~mask])
                plt.plot(c_wavelengths[~mask], actual[~mask])
                plt.plot(c_wavelengths[~mask], rfr_delta[~mask])
            if not no_plot:
                plt.plot(c_wavelengths, [0]*len(c_wavelengths))
            err_term = np.sum(np.power(rfr_delta[~delta_mask], 2))/len(c_wavelengths[~delta_mask])
            err_sum = np.sum(rfr_delta[~delta_mask])/len(rfr_delta[~delta_mask])
            red_chi = np.sum(np.power(rfr_delta[~delta_mask], 2)*ivar[~delta_mask])/(len(c_wavelengths[~delta_mask])-var_count-1)
            if not no_plot:
                plt.legend(['Predicted', 'Actual', 'Delta {:0.5f}'.format(err_term)])
                plt.tight_layout()
                plt.title("Random Forest Regressor: {}".format(title_str))
                plt.show()
                plt.close()
            print err_term, red_chi, err_sum,

            if include_knn:
                knn_prediction = knn.predict(test_X)
                if not use_spca and not use_pca:
                    knn_predicted = ica.inverse_transform(knn_prediction, copy=True)
                else:
                    knn_predicted = np.zeros( (1, ica.components_.shape[1]) )
                    knn_predicted[0,:] = np.sum(knn_prediction.T * ica.components_, 0)

                if use_filter_split:
                    knn_e_prediction = knn_e.predict(test_X)
                    if not use_spca and not use_pca:
                        knn_e_predicted = ica_e.inverse_transform(knn_e_prediction, copy=True)
                    else:
                        knn_e_predicted = np.zeros( (1, ica_e.components_.shape[1]) )
                        knn_e_predicted[0,:] = np.sum(knn_e_prediction.T * ica_e.components_, 0)
                    knn_predicted = knn_predicted + knn_e_predicted

                if not no_plot:
                    plt.plot(c_wavelengths[~mask], knn_predicted[0][~mask])
                    plt.plot(c_wavelengths[~mask], actual[~mask])
                knn_delta = knn_predicted[0] - actual
                err_term = np.sum(np.power(knn_delta[~delta_mask], 2))/len(c_wavelengths[~delta_mask])
                err_sum = np.sum(knn_delta[~delta_mask])/len(knn_delta[~delta_mask])
                red_chi = np.sum(np.power(knn_delta[~delta_mask], 2)*ivar[~delta_mask])/(len(c_wavelengths[~delta_mask])-var_count-1)

                if not no_plot:
                    plt.plot(c_wavelengths[~mask], knn_delta[~mask])
                    plt.plot(c_wavelengths, [0]*len(c_wavelengths))
                    plt.legend(['Predicted', 'Actual', 'Delta {:0.5f}'.format(err_term)])
                    plt.tight_layout()
                    plt.title("Good 'ol K-NN: {}".format(title_str))
                    plt.show()
                    plt.close()
                print err_term, red_chi, err_sum,

        if include_linear:
            poly_1_prediction = linear.predict(test_X)
            if not use_spca and not use_pca:
                poly_1_predicted = ica.inverse_transform(poly_1_prediction, copy=True)
            else:
                poly_1_predicted = np.zeros( (1, ica.components_.shape[1]) )
                poly_1_predicted[0,:] = np.sum(poly_1_prediction.T * ica.components_, 0)

            if use_filter_split:
                poly_1_e_prediction = linear.predict(test_X)
                if not use_spca and not use_pca:
                    poly_1_e_predicted = ica_e.inverse_transform(poly_1_e_prediction, copy=True)
                else:
                    poly_1_e_predicted = np.zeros( (1, ica_e.components_.shape[1]) )
                    poly_1_e_predicted[0,:] = np.sum(poly_1_e_prediction.T * ica_e.components_, 0)
                poly_1_predicted = poly_1_predicted + poly_1_e_predicted

            poly_1_delta = poly_1_predicted[0] - actual

            if not no_plot:
                plt.plot(c_wavelengths[~mask], poly_1_predicted[0][~mask])
                plt.plot(c_wavelengths[~mask], actual[~mask])
            err_term = np.sum(np.power(poly_1_delta[~delta_mask], 2))/len(c_wavelengths[~delta_mask])
            err_sum = np.sum(poly_1_delta[~delta_mask])/len(poly_1_delta[~delta_mask])
            red_chi = np.sum(np.power(poly_1_delta[~delta_mask], 2)*ivar[~delta_mask])/(len(c_wavelengths[~delta_mask])-var_count-1)

            if not no_plot:
                plt.plot(c_wavelengths[~mask], poly_1_delta[~mask])
                plt.plot(c_wavelengths, [0]*len(c_wavelengths))
                plt.legend(['Predicted', 'Actual', 'Delta {:0.5f}'.format(err_term)])
                plt.tight_layout()
                plt.title("Poly 1: {}".format(title_str))
                plt.show()
                plt.close()

            print err_term, red_chi, err_sum,

            poly_2_prediction = poly_2_linear.predict(test_X)
            if not use_spca and not use_pca:
                poly_2_predicted = ica.inverse_transform(poly_2_prediction, copy=True)
            else:
                poly_2_predicted = np.zeros( (1, ica.components_.shape[1]) )
                poly_2_predicted[0,:] = np.sum(poly_2_prediction.T * ica.components_, 0)

            poly_2_delta = poly_2_predicted[0] - actual

            if not no_plot:
                plt.plot(c_wavelengths[~mask], poly_2_predicted[0][~mask])
                plt.plot(c_wavelengths[~mask], actual[~mask])
            err_term = np.sum(np.power(poly_2_delta[~delta_mask], 2))/len(c_wavelengths[~delta_mask])
            err_sum = np.sum(poly_2_delta[~delta_mask])/len(poly_2_delta[~delta_mask])
            red_chi = np.sum(np.power(poly_2_delta[~delta_mask], 2)*ivar[~delta_mask])/(len(c_wavelengths[~delta_mask])-var_count-1)

            if not no_plot:
                plt.plot(c_wavelengths[~mask], poly_2_delta[~mask])
                plt.plot(c_wavelengths, [0]*len(c_wavelengths))
                plt.legend(['Predicted', 'Actual', 'Delta {:0.5f}'.format(err_term)])
                plt.tight_layout()
                plt.title("Poly 2: {}".format(title_str))
                plt.show()
                plt.close()

            print err_term, red_chi, err_sum,
            err_ind =+ 1

            if order_3:
                poly_3_prediction = poly_3_linear.predict(test_X)
                if not use_spca and not use_pca:
                    poly_3_predicted = ica.inverse_transform(poly_3_prediction, copy=True)
                else:
                    poly_3_predicted = np.zeros( (1, ica.components_.shape[1]) )
                    poly_3_predicted[0,:] = np.sum(poly_3_prediction.T * ica.components_, 0)

                poly_3_delta = poly_3_predicted[0] - actual

                if not no_plot:
                    plt.plot(c_wavelengths[~mask], poly_3_predicted[0][~mask])
                    plt.plot(c_wavelengths[~mask], actual[~mask])
                err_term = np.sum(np.power(poly_3_delta[~delta_mask], 2))/len(c_wavelengths[~delta_mask])
                err_sum = np.sum(poly_3_delta[~delta_mask])/len(poly_3_delta[~delta_mask])
                red_chi = np.sum(np.power(poly_3_delta[~delta_mask], 2)*ivar[~delta_mask])/(len(c_wavelengths[~delta_mask])-var_count-1)

                if not no_plot:
                    plt.plot(c_wavelengths[~mask], poly_3_delta[~mask])
                    plt.plot(c_wavelengths, [0]*len(c_wavelengths))
                    plt.legend(['Predicted', 'Actual', 'Delta {:0.5f}'.format(err_term)])
                    plt.tight_layout()
                    plt.title("Poly 3: {}".format(title_str))
                    plt.show()
                    plt.close()

                print err_term, red_chi, err_sum,
                err_ind =+ 1

            if order_4:
                poly_4_prediction = poly_4_linear.predict(test_X)
                if not use_spca and not use_pca:
                    poly_4_predicted = ica.inverse_transform(poly_4_prediction, copy=True)
                else:
                    poly_4_predicted = np.zeros( (1, ica.components_.shape[1]) )
                    poly_4_predicted[0,:] = np.sum(poly_4_prediction.T * ica.components_, 0)

                poly_4_delta = poly_4_predicted[0] - actual

                if not no_plot:
                    plt.plot(c_wavelengths[~mask], poly_4_predicted[0][~mask])
                    plt.plot(c_wavelengths[~mask], actual[~mask])
                err_term = np.sum(np.power(poly_4_delta[~delta_mask], 2))/len(c_wavelengths[~delta_mask])
                err_sum = np.sum(poly_4_delta[~delta_mask])/len(poly_4_delta[~delta_mask])
                red_chi = np.sum(np.power(poly_4_delta[~delta_mask], 2)*ivar[~delta_mask])/(len(c_wavelengths[~delta_mask])-var_count-1)

                if not no_plot:
                    plt.plot(c_wavelengths[~mask], poly_4_delta[~mask])
                    plt.plot(c_wavelengths, [0]*len(c_wavelengths))
                    plt.legend(['Predicted', 'Actual', 'Delta {:0.5f}'.format(err_term)])
                    plt.tight_layout()
                    plt.title("Poly 4: {}".format(title_str))
                    plt.show()
                    plt.close()

                print err_term, red_chi, err_sum,
                err_ind =+ 1

        print

        if save_out:
            out_table = Table()
            wavelength_col = Column(c_wavelengths, name="wavelength", dtype=float)
            out_table.add_columns([wavelength_col])

            if not linear_only:
                rf_col = Column(rfr_predicted[0], name="rf_flux", dtype=float)
                out_table.add_columns([rf_col])

                if include_knn:
                    knn_col = Column(knn_predicted[0], name="knn_flux", dtype=float)
                    avg_col = Column(avg_predicted[0], name="avg_flux", dtype=float)
                    out_table.add_columns([knn_col, avg_col])

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

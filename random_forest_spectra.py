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
from sklearn import preprocessing as skpp
from sklearn.pipeline import make_pipeline
from sklearn.cross_decomposition import PLSRegression

import ICAize

import fnmatch
import os
import os.path
import sys

rfr_random_state = 456371

def load_all_spectra_data(path, noncon_file=None, con_file=None):
    nc_sources, nc_mixing, noncon_exposures = load_spectra_data(path,
                                target_type='noncontinuum', filename=noncon_file)
    c_sources, c_mixing, c_exposures = load_spectra_data(path,
                                target_type='continuum', filename=con_file)

    return c_sources, c_mixing, c_exposures, nc_sources, nc_mixing, noncon_exposures

def load_spectra_data(path, target_type='continuum', filename=None):
    if filename is None:
        filename = ICAize.data_file.format(target_type)
    #print "Loading:", os.path.join(path, filename)
    npz = np.load(os.path.join(path, filename))

    sources = npz['sources']
    mixing = npz['mixing']
    exposures = npz['exposures']
    wavelengths = npz['wavelengths']

    npz.close()
    #print "Using", len(sources), "sources with length", len(sources[0])
    #print mixing.shape
    #print exposures.shape
    return sources, mixing, exposures, wavelengths

def load_observation_metadata(path, file = "annotated_metadata.csv"):
    data = Table.read(os.path.join(path, file), format="ascii.csv")

    return data

def trim_observation_metadata(data, copy=False):
    if copy:
        data = data.copy()

    kept_columns = ['EXP_ID', 'RA', 'DEC', 'AZ', 'ALT', 'AIRMASS',
                    'LUNAR_MAGNITUDE', 'LUNAR_ELV', 'LUNAR_SEP', 'SOLAR_ELV',
                    'SOLAR_SEP', 'GALACTIC_CORE_SEP', 'GALACTIC_PLANE_SEP', 'SS_COUNT', 'SS_AREA']
    removed_columns = [name for name in data.colnames if name not in kept_columns]
    data.remove_columns(removed_columns)

    return data

def main():
    metadata_path = ".."
    spectra_path = "."
    test_ind = 0
    #train_span = slice(None,-1,None)
    target_types = 'continuum'

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

    results = None
    wavelengths = None
    for target_type in target_types:
        for test_ind in test_inds:
            wavelengths, rfr_result, knn_result, errs = load_plot_etc_target_type(metadata_path, spectra_path, int(test_ind), target_type, no_plot=True)
            if results is None:
                results = [rfr_result, knn_result, errs, np.empty((3,), dtype=float)]
            else:
                results[0] += rfr_result
                results[1] += knn_result
                results[2] += errs
                results[3] += np.power(errs,2)
    '''
    for result in results:
        plt.plot(wavelengths, result)
        plt.plot(wavelengths, [0]*len(wavelengths))
        plt.legend(['Total Delta'])
        plt.tight_layout()
        plt.show()
        plt.close()
    '''

    print results[2]/len(test_ind)
    print results[3]/len(test_ind)-np.power(results[2]/len(test_ind))


def load_plot_etc_target_type(metadata_path, spectra_path, test_ind, target_type, no_plot=False):
    obs_metadata = trim_observation_metadata(load_observation_metadata(metadata_path))
    #c_sources, c_mixing, c_exposures, nc_sources, nc_mixing, noncon_exposures = load_all_spectra_data(spectra_path)
    c_sources, c_mixing, c_exposures, c_wavelengths = load_spectra_data(spectra_path, target_type=target_type)
    #nc_sources, nc_mixing, nc_exposures, nc_wavelengths = load_spectra_data(spectra_path, target_type='noncontinuum')

    reduced_obs_metadata = obs_metadata[np.in1d(obs_metadata['EXP_ID'], c_exposures)]
    reduced_obs_metadata.sort('EXP_ID')
    sorted_inds = np.argsort(c_exposures)

    errs = np.empty((3,), dtype=float)

    rfr = ensemble.RandomForestRegressor(n_estimators=200, min_samples_split=1,
                        random_state=rfr_random_state, n_jobs=-1, verbose=False)
    #mtencv = linear_model.MultiTaskElasticNetCV(copy_X=True, normalize=False, n_alphas=200, n_jobs=-1)
    #random_state=rfr_random_state, selection='random',
    #rnn = neighbors.RadiusNeighborsRegressor(weights='distance')
    knn = neighbors.KNeighborsRegressor(weights='distance', n_neighbors=10, p=64)
    #ransac = make_pipeline(skpp.PolynomialFeatures(3), linear_model.RANSACRegressor(random_state=rfr_random_state))
    #omp = make_pipeline(skpp.PolynomialFeatures(3), linear_model.OrthogonalMatchingPursuit())
    #pls = PLSRegression(n_components=12, scale = False, max_iter=750)

    reduced_obs_metadata.remove_column('EXP_ID')
    md_len = len(reduced_obs_metadata)
    X_arr = np.array(reduced_obs_metadata).view('f8').reshape((md_len,-1))


    ################################################################


    test_X = X_arr[test_ind]
    train_X = np.vstack( [X_arr[:test_ind], X_arr[test_ind+1:]] )
    test_y =  (c_sources[sorted_inds])[test_ind]
    train_y = np.vstack( [(c_sources[sorted_inds])[:test_ind], (c_sources[sorted_inds])[test_ind+1:]] )

    #print "Which is exposure:", c_exposures[sorted_inds[test_ind]]
    title_str = "exp{}, {}".format(c_exposures[sorted_inds[test_ind]], target_type)

    ica = ICAize.unpickle_FastICA(target_type=target_type)

    rfr.fit(X=train_X, y=train_y)
    #mtencv.fit(X=train_X, y=train_y)
    #rnn.fit(X=train_X, y=train_y)
    knn.fit(X=train_X, y=train_y)
    #ransac.fit(X=train_X, y=train_y)
    #pls.fit(X=train_X, Y=train_y)

    rfr_prediction = rfr.predict(test_X)
    rfr_predicted_continuum = ica.inverse_transform(rfr_prediction, copy=True)

    print test_ind

    data = None
    actual = None
    mask = None
    for file in os.listdir(spectra_path):
        if fnmatch.fnmatch(file, "stacked_sky_*exp{}-continuum.csv".format(c_exposures[sorted_inds[test_ind]])):
            data = Table.read(os.path.join(spectra_path, file), format="ascii.csv")
            mask = (data['ivar'] == 0) | np.isclose(rfr_predicted_continuum[0], 0)
            #data['con_flux'][mask] = np.interp(data['wavelength'][mask], data['wavelength'][~mask], data['con_flux'][~mask])

            actual = data['flux']
            if target_type == 'continuum':
                actual = data['con_flux']

            #print rfr_predicted_continuum[0][~mask]

            rfr_delta = rfr_predicted_continuum[0] - actual
            if not no_plot:
                plt.plot(c_wavelengths[~mask], rfr_predicted_continuum[0][~mask])
                plt.plot(c_wavelengths[~mask], actual[~mask])
                plt.plot(c_wavelengths[~mask], rfr_delta[~mask])
    if not no_plot:
        plt.plot(c_wavelengths, [0]*len(c_wavelengths))
    err_term = np.sum(np.power(rfr_delta[~mask], 2))/len(c_wavelengths[~mask])
    if not no_plot:
        plt.legend(['Predicted', 'Actual', 'Delta {:0.5f}'.format(err_term)])
        plt.tight_layout()
        plt.title("Random Forest Regressor: {}".format(title_str))
        plt.show()
        plt.close()
    print err_term,
    errs[0] = err_term

    knn_prediction = knn.predict(test_X)
    knn_predicted_continuum = ica.inverse_transform(knn_prediction, copy=True)

    if not no_plot:
        plt.plot(c_wavelengths[~mask], knn_predicted_continuum[0][~mask])
        plt.plot(c_wavelengths[~mask], actual[~mask])
    knn_delta = knn_predicted_continuum[0] - actual
    err_term = np.sum(np.power(knn_delta[~mask], 2))/len(c_wavelengths[~mask])
    if not no_plot:
        plt.plot(c_wavelengths[~mask], knn_delta[~mask])
        plt.plot(c_wavelengths, [0]*len(c_wavelengths))
        plt.legend(['Predicted', 'Actual', 'Delta {:0.5f}'.format(err_term)])
        plt.tight_layout()
        plt.title("Good 'ol K-NN: {}".format(title_str))
        plt.show()
        plt.close()
    print err_term,
    errs[1] = err_term

    avg_predicted_continuum = (knn_predicted_continuum + rfr_predicted_continuum)/2

    if not no_plot:
        plt.plot(c_wavelengths[~mask], avg_predicted_continuum[0][~mask])
        plt.plot(c_wavelengths[~mask], actual[~mask])
    avg_delta = avg_predicted_continuum[0] - actual
    err_term = np.sum(np.power(avg_delta[~mask], 2))/len(c_wavelengths[~mask])
    if not no_plot:
        plt.plot(c_wavelengths[~mask], avg_delta[~mask])
        plt.plot(c_wavelengths, [0]*len(c_wavelengths))
        plt.legend(['Predicted', 'Actual', 'Delta {:0.5f}'.format(err_term)])
        plt.tight_layout()
        plt.title("Avg. of KNN,RFR: {}".format(title_str))
        plt.show()
        plt.close()
    print err_term
    errs[2] = err_term

    return c_wavelengths, rfr_delta, knn_delta, errs


if __name__ == '__main__':
    main()

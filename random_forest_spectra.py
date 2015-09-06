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
    print "Loading:", os.path.join(path, filename)
    npz = np.load(os.path.join(path, filename))

    sources = npz['sources']
    mixing = npz['mixing']
    exposures = npz['exposures']
    wavelengths = npz['wavelengths']

    npz.close()
    print "Using", len(sources), "sources"
    return sources, mixing, exposures, wavelengths

def load_observation_metadata(path, file = "annotated_metadata.csv"):
    data = Table.read(os.path.join(path, file), format="ascii.csv")

    return data

def trim_observation_metadata(data, copy=False):
    if copy:
        data = data.copy()

    kept_columns = ['EXP_ID', 'RA', 'DEC', 'AZ', 'ALT', 'AIRMASS',
                    'LUNAR_MAGNITUDE', 'LUNAR_ELV', 'LUNAR_SEP', 'SOLAR_ELV',
                    'SOLAR_SEP', 'GALACTIC_CORE_SEP', 'GALACTIC_PLANE_SEP']
    removed_columns = [name for name in data.colnames if name not in kept_columns]
    data.remove_columns(removed_columns)

    return data

def main():
    metadata_path = ".."
    spectra_path = "."
    test_ind = 0
    #train_span = slice(None,-1,None)
    if len(sys.argv) == 2:
        metadata_path = sys.argv[1]
    if len(sys.argv) == 3:
        metadata_path = sys.argv[1]
        spectra_path = sys.argv[2]
    if len(sys.argv) == 4:
        metadata_path = sys.argv[1]
        spectra_path = sys.argv[2]
        test_ind = int(sys.argv[3])

    obs_metadata = trim_observation_metadata(load_observation_metadata(metadata_path))
    #c_sources, c_mixing, c_exposures, nc_sources, nc_mixing, noncon_exposures = load_all_spectra_data(spectra_path)
    c_sources, c_mixing, c_exposures, c_wavelengths = load_spectra_data(spectra_path, target_type='noncontinuum')
    #nc_sources, nc_mixing, nc_exposures, nc_wavelengths = load_spectra_data(spectra_path, target_type='noncontinuum')

    reduced_obs_metadata = obs_metadata[np.in1d(obs_metadata['EXP_ID'], c_exposures)]
    reduced_obs_metadata.sort('EXP_ID')
    sorted_inds = np.argsort(c_exposures)

    rfr = ensemble.RandomForestRegressor(n_estimators=150, min_samples_split=1,
                        random_state=rfr_random_state, n_jobs=-1, verbose=True)
    #mtencv = linear_model.MultiTaskElasticNetCV(copy_X=True, normalize=False, n_alphas=200, n_jobs=-1)
    #random_state=rfr_random_state, selection='random',
    #rnn = neighbors.RadiusNeighborsRegressor(weights='distance')
    knn = neighbors.KNeighborsRegressor(weights='distance', n_neighbors=1x5, p=128)

    reduced_obs_metadata.remove_column('EXP_ID')
    md_len = len(reduced_obs_metadata)
    X_arr = np.array(reduced_obs_metadata).view('f8').reshape((md_len,-1))


    ################################################################


    test_X = X_arr[test_ind]
    train_X = np.vstack( [X_arr[:test_ind], X_arr[test_ind+1:]] )
    test_y =  (c_sources[sorted_inds])[test_ind]
    train_y = np.vstack( [(c_sources[sorted_inds])[:test_ind], (c_sources[sorted_inds])[test_ind+1:]] )

    print "Which is exposure:", c_exposures[sorted_inds[test_ind]]

    ica = ICAize.unpickle_FastICA(target_type='noncontinuum')

    rfr.fit(X=train_X, y=train_y)
    #mtencv.fit(X=train_X, y=train_y)
    #rnn.fit(X=train_X, y=train_y)
    knn.fit(X=train_X, y=train_y)

    prediction = rfr.predict(test_X)
    predicted_continuum = ica.inverse_transform(prediction, copy=True)

    data = None
    actual = None
    plt.plot(c_wavelengths, predicted_continuum[0])
    for file in os.listdir(spectra_path):
        if fnmatch.fnmatch(file, "stacked_sky_*exp{}-continuum.csv".format(c_exposures[sorted_inds[test_ind]])):
            data = Table.read(os.path.join(spectra_path, file), format="ascii.csv")
            mask = data['ivar'] == 0
            data['con_flux'][mask] = np.interp(data['wavelength'][mask], data['wavelength'][~mask], data['con_flux'][~mask])
            #actual = data['con_flux']
            actual = data['flux']
            delta = predicted_continuum[0] - actual
            plt.plot(c_wavelengths, actual)
            plt.plot(c_wavelengths, delta)
    plt.plot(c_wavelengths, [0]*len(c_wavelengths))
    err_term = np.sum(np.power(delta, 2))/len(c_wavelengths)
    plt.legend(['Predicted', 'Actual', 'Delta {:0.5f}'.format(err_term)])
    plt.tight_layout()
    plt.title("Random Forest Regressor")
    plt.show()
    plt.close()


    '''
    prediction = mtencv.predict(test_X)
    predicted_continuum = ica.inverse_transform(prediction, copy=True)

    plt.plot(c_wavelengths, predicted_continuum)
    plt.plot(c_wavelengths, actual)
    delta = predicted_continuum - actual
    err_term = np.sum(np.power(delta, 2))/len(c_wavelengths)
    plt.plot(c_wavelengths, delta)
    plt.plot(c_wavelengths, [0]*len(c_wavelengths))
    plt.legend(['Predicted', 'Actual', 'Delta {:0.5f}'.format(err_term)])
    plt.tight_layout()
    plt.title("MultiTask Elastic Net CV")
    plt.show()
    plt.close()
    '''


    #np.seterr(all='raise')
    '''
    prediction = rnn.predict(test_X)
    print prediction

    predicted_continuum = ica.inverse_transform(prediction, copy=True)
    print predicted_continuum.shape
    print c_wavelengths.shape

    plt.plot(c_wavelengths, predicted_continuum[0])
    plt.plot(c_wavelengths, actual)
    plt.plot(c_wavelengths, predicted_continuum[0] - actual)
    plt.plot(c_wavelengths, [0]*len(c_wavelengths))
    plt.legend(['Predicted', 'Actual', 'Delta'])
    plt.tight_layout()
    plt.show()
    plt.close()
    '''


    prediction = knn.predict(test_X)
    predicted_continuum = ica.inverse_transform(prediction, copy=True)

    plt.plot(c_wavelengths, predicted_continuum[0])
    plt.plot(c_wavelengths, actual)
    delta = predicted_continuum[0] - actual
    err_term = np.sum(np.power(delta, 2))/len(c_wavelengths)
    plt.plot(c_wavelengths, delta)
    plt.plot(c_wavelengths, [0]*len(c_wavelengths))
    plt.legend(['Predicted', 'Actual', 'Delta {:0.5f}'.format(err_term)])
    plt.tight_layout()
    plt.title("Good 'ol K-Nearest")
    plt.show()
    plt.close()

if __name__ == '__main__':
    main()

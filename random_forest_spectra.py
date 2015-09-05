import numpy as np
import matplotlib.pyplot as plt

from astropy.table import Table
from astropy.table import Column
from astropy.table import vstack
from astropy.table import join

from sklearn import ensemble
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
    if len(sys.argv) == 2:
        metadata_path = sys.argv[1]
    if len(sys.argv) == 3:
        metadata_path = sys.argv[1]
        spectra_path = sys.argv[2]

    obs_metadata = trim_observation_metadata(load_observation_metadata(metadata_path))
    #c_sources, c_mixing, c_exposures, nc_sources, nc_mixing, noncon_exposures = load_all_spectra_data(spectra_path)
    c_sources, c_mixing, c_exposures, c_wavelengths = load_spectra_data(spectra_path) #, target_type='noncontinuum')
    #nc_sources, nc_mixing, nc_exposures, nc_wavelengths = load_spectra_data(spectra_path, target_type='noncontinuum')

    reduced_obs_metadata = obs_metadata[np.in1d(obs_metadata['EXP_ID'], c_exposures)]
    reduced_obs_metadata.sort('EXP_ID')
    sorted_inds = np.argsort(c_exposures)

    rfr = ensemble.RandomForestRegressor(n_estimators=150, min_samples_split=1, random_state=rfr_random_state)

    reduced_obs_metadata.remove_column('EXP_ID')
    md_len = len(reduced_obs_metadata)
    X_arr = np.array(reduced_obs_metadata).view('f8').reshape((md_len,-1))

    '''
    print "--------------------Trainding data-------------------"
    print X_arr[:-1]
    print (c_sources[sorted_inds])[-1]
    print "--------------------Candidate data-------------------"
    print X_arr[-1]
    print (c_sources[sorted_inds])[-1]
    '''
    test_ind = 0
    train_span = slice(0,None,None)

    print "Which is exposure:", c_exposures[sorted_inds[test_ind]]

    rfr.fit(X=X_arr[train_span], y=(c_sources[sorted_inds])[train_span])
    #rfr.fit(X=X_arr[:], y=(c_sources[sorted_inds])[:])
    print rfr.feature_importances_

    prediction = rfr.predict(X_arr[test_ind])
    print prediction

    ica = ICAize.unpickle_FastICA() #(target_type='noncontinuum')
    predicted_continuum = ica.inverse_transform(prediction, copy=True)
    print predicted_continuum.shape
    print c_wavelengths.shape

    plt.plot(c_wavelengths, predicted_continuum[0])
    for file in os.listdir(spectra_path):
        if fnmatch.fnmatch(file, "stacked_sky_*exp{}-continuum.csv".format(c_exposures[sorted_inds[test_ind]])):
            data = Table.read(os.path.join(spectra_path, file), format="ascii.csv")
            mask = data['ivar'] == 0
            data['con_flux'][mask] = np.interp(data['wavelength'][mask], data['wavelength'][~mask], data['con_flux'][~mask])
            actual = data['con_flux']
            #actual = data['flux']
            plt.plot(c_wavelengths, actual)
            plt.plot(c_wavelengths, predicted_continuum[0] - actual)
    plt.plot(c_wavelengths, [0]*len(c_wavelengths))
    plt.legend(['Predicted', 'Actual', 'Delta'])
    plt.tight_layout()
    plt.show()
    plt.close()

if __name__ == '__main__':
    main()

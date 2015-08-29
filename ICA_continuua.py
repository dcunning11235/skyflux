import numpy as np
import matplotlib.pyplot as plt

from astropy.table import Table
from astropy.table import Column
from astropy.table import vstack
from astropy.table import join

from sklearn.decomposition import FastICA
from sklearn import preprocessing as skpp

import fnmatch
import os
import os.path
import sys

import gradient_boost_peaks as gbk

def load_all_in_dir(path, use_con_flux=True, recombine_flux=False):
    pattern = "stacked*-continuum.csv"
    flux_list = []
    exp_list = []
    wavelengths = None

    for file in os.listdir(path):
        if fnmatch.fnmatch(file, pattern):
            data = Table.read(os.path.join(path, file), format="ascii.csv")
            exp = int(file.split("-")[2][3:])
            exp_col = Column([exp]*len(data), name="EXP_ID")

            if wavelengths is None:
                wavelengths = np.array(data['wavelength'], copy=False)

            if not recombine_flux:
                y_col = 'con_flux' if use_con_flux else 'flux'
                flux_list.append(np.array(data[y_col], copy=False))
            else:
                flux_list.append(np.array(data['con_flux'] + data['flux'], copy=False))
            exp_list.append(exp)

    flux_arr = np.array(flux_list)

    return flux_arr, exp_list, wavelengths

def get_X_and_y(data_subset, obs_metadata, use_con_flux=True):
    # X: n_samples, n_features
    # y: n_samples
    # E.g., for this case, X will be the observation metadata for each exposure and y will be
    #       the total flux for a given line/peak (actually x2:  total_flux and total_con_flux)
    y_col = 'con_flux' if use_con_flux else 'flux'
    # Need to 'de-peak' flux if going to do this... perhaps.  Well, this should be an option
    # in any case
    full_table = join(obs_metadata, data_subset['EXP_ID', y_col])

    labels = full_table['EXP_ID']
    full_table.remove_column('EXP_ID')
    y = full_table[y_col]
    full_table.remove_column(y_col)

    return full_table, y, labels

def main():
    path = "."
    if len(sys.argv) == 2:
        path = sys.argv[1]
    flux_arr, exposure_list, wavelengths = load_all_in_dir(path, use_con_flux=True, recombine_flux=False)
    #obs_metadata = gbk.trim_observation_metadata(gbk.load_observation_metadata(path))
    ica = FastICA(n_components = 20, whiten=True, max_iter=500, fun='exp', random_state=1234975)
    eigen_spectra_comps = ica.fit(flux_arr).transform(flux_arr)

    print eigen_spectra_comps
    print eigen_spectra_comps.shape

    print ica.components_
    print ica.transform(flux_arr[0])

if __name__ == '__main__':
    main()

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
import random
import pickle

ica_max_iter=750
ica_random_state=1234975
ica_continuum_n=70
ica_noncontinuum_n=110

data_file = "{}_sources_and_mixing.npz"
pickle_file = "fastica_{}_pickle.pkl"

def main():
    path = "."
    mixing_matrix_path = None
    if len(sys.argv) == 2:
        path = sys.argv[1]
    if len(sys.argv) == 3:
        mixing_matrix_path = sys.argv[2]

    con_flux_arr, con_exposure_arr, con_wavelengths = load_all_in_dir(path,
                                                    use_con_flux=True, recombine_flux=False)
    flux_arr, exposure_arr, wavelengths = load_all_in_dir(path, use_con_flux=False,
                                                    recombine_flux=False)

    con_sources, con_mixing, con_model = reduce_with_ica(con_flux_arr, ica_continuum_n)
    noncon_sources, noncon_mixing, noncon_model = reduce_with_ica(flux_arr, ica_noncontinuum_n)

    np.savez(data_file.format("continuum"), sources=con_sources, mixing=con_mixing,
                exposures=con_exposure_arr, wavelengths=wavelengths)
    pickle_FastICA(con_model)

    np.savez(data_file.format("noncontinuum"), sources=noncon_sources, mixing=noncon_mixing,
                exposures=exposure_arr, wavelengths=wavelengths)
    pickle_FastICA(noncon_model, target_type='noncontinuum')


def pickle_FastICA(model, path='.', target_type='continuum', filename=None):
    if filename is None:
        filename = pickle_file.format(target_type)
    output = open(os.path.join(path, filename), 'wb')
    pickle.dump(model, output)
    output.close()

def unpickle_FastICA(path='.', target_type='continuum', filename=None):
    if filename is None:
        filename = "fastica_{}_pickle.pkl".format(target_type)
    output = open(os.path.join(path, filename), 'rb')
    model = pickle.load(output)
    output.close()

    return model

def get_FastICA(target_type='continuum', n=None, mixing=None):
    if n is not None:
        return FastICA(n_components = n, whiten=True, max_iter=ica_max_iter,
                        random_state=ica_random_state, w_init=mixing)
    else:
        if target_type == 'continuum':
            ret = FastICA(n_components = ica_continuum_n, whiten=True, max_iter=ica_max_iter,
                        random_state=ica_random_state, w_init=mixing)
            print "Mixing:", mixing
            print "Init'd FastICA with mixing:", ret.mixing_
            return ret
        elif target_type == 'noncontinuum':
            return FastICA(n_components = ica_noncontinuum_n, whiten=True, max_iter=ica_max_iter,
                        random_state=ica_random_state, w_init=mixing)

    return None

def reduce_with_ica(flux_arr, n, mixing=None):
    ica = get_FastICA(n=n, mixing=mixing)
    if mixing is None:
        sources = ica.fit(flux_arr)

    sources = ica.transform(flux_arr, copy=True)

    return sources, ica.mixing_, ica

def load_all_in_dir(path, use_con_flux=True, recombine_flux=False):
    pattern = "stacked*-continuum.csv"
    flux_list = []
    exp_list = []
    wavelengths = None

    for file in os.listdir(path):
        if fnmatch.fnmatch(file, pattern):
            data = Table(Table.read(os.path.join(path, file), format="ascii.csv"), masked=True)
            mask = data['ivar'] == 0

            data['con_flux'][mask] = np.interp(data['wavelength'][mask], data['wavelength'][~mask], data['con_flux'][~mask])

            exp = int(file.split("-")[2][3:])

            if wavelengths is None:
                wavelengths = np.array(data['wavelength'], copy=False)

            if not recombine_flux:
                y_col = 'con_flux' if use_con_flux else 'flux'
                flux_list.append(np.array(data[y_col], copy=False))
            else:
                flux_list.append(np.array(data['con_flux'] + data['flux'], copy=False))
            exp_list.append(exp)

    flux_arr = np.array(flux_list)
    exp_arr = np.array(exp_list)

    return flux_arr, exp_arr, wavelengths

if __name__ == '__main__':
    main()
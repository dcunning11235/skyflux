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

    con_flux_arr, con_exposure_arr, con_masks, con_wavelengths = load_all_in_dir(path,
                                                    use_con_flux=True, recombine_flux=False)
    flux_arr, exposure_arr, masks, wavelengths = load_all_in_dir(path, use_con_flux=False,
                                                    recombine_flux=False)
    comb_flux_arr, comb_exposure_arr, comb_masks, comb_wavelengths = \
        con_flux_arr + flux_arr, con_exposure_arr[:], con_masks[:], con_wavelengths[:]

    mask_summed = np.sum(con_masks, axis=0)

    min_val_ind = np.min(np.where(mask_summed == 0))
    max_val_ind = np.max(np.where(mask_summed == 0))
    print min_val_ind, max_val_ind

    for i in range(con_flux_arr.shape[0]):
        con_flux_arr[i,:min_val_ind] = 0
        con_flux_arr[i,max_val_ind+1:] = 0

    con_sources, con_mixing, con_model = reduce_with_ica(con_flux_arr, ica_continuum_n)
    noncon_sources, noncon_mixing, noncon_model = reduce_with_ica(flux_arr, ica_noncontinuum_n)
    comb_sources, comb_mixing, comb_model = reduce_with_ica(comb_flux_arr, ica_noncontinuum_n)

    np.savez(data_file.format("continuum"), sources=con_sources, mixing=con_mixing,
                exposures=con_exposure_arr, wavelengths=wavelengths)
    pickle_FastICA(con_model)

    np.savez(data_file.format("noncontinuum"), sources=noncon_sources, mixing=noncon_mixing,
                exposures=exposure_arr, wavelengths=wavelengths)
    pickle_FastICA(noncon_model, target_type='noncontinuum')

    np.savez(data_file.format("combined"), sources=comb_sources, mixing=comb_mixing,
                exposures=exposure_arr, wavelengths=wavelengths)
    pickle_FastICA(comb_model, target_type='combined')

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
            return FastICA(n_components = ica_continuum_n, whiten=True, max_iter=ica_max_iter,
                        random_state=ica_random_state, w_init=mixing)
        elif target_type == 'noncontinuum':
            return FastICA(n_components = ica_noncontinuum_n, whiten=True, max_iter=ica_max_iter,
                        random_state=ica_random_state, w_init=mixing)
        elif target_type == 'combined':
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
    mask_list = []
    wavelengths = None

    for file in os.listdir(path):
        if fnmatch.fnmatch(file, pattern):
            data = Table(Table.read(os.path.join(path, file), format="ascii.csv"), masked=True)
            mask = data['ivar'] == 0

            exp = int(file.split("-")[2][3:])

            if wavelengths is None:
                wavelengths = np.array(data['wavelength'], copy=False)

            if not recombine_flux:
                y_col = 'con_flux' if use_con_flux else 'flux'
                flux_list.append(np.array(data[y_col], copy=False))
            else:
                flux_list.append(np.array(data['con_flux'] + data['flux'], copy=False))
            mask_list.append(mask)
            exp_list.append(exp)

    flux_arr = np.array(flux_list)
    exp_arr = np.array(exp_list)
    mask_arr = np.array(mask_list)

    return flux_arr, exp_arr, mask_arr, wavelengths

if __name__ == '__main__':
    main()

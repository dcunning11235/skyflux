import numpy as np
import matplotlib.pyplot as plt

from astropy.table import Table
from astropy.table import Column
from astropy.table import vstack
from astropy.table import join

from sklearn.decomposition import FastICA
from sklearn.decomposition import SparsePCA
from sklearn import preprocessing as skpp

import fnmatch
import os
import os.path
import sys
import random
import pickle

ica_max_iter=750
spca_max_iter = 2500

ica_random_state=1234975
ica_continuum_n=70
ica_noncontinuum_n=110

ica_data_file = "fastica_{}_sources_and_mixing.npz"
spca_data_file = "spca_{}_souces_and_components.npz"
ica_pickle_file = "fastica_{}_pickle.pkl"
spca_pickle_file = "spca_{}_pickle.pkl"

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

    def _ica_reduce_and_save(flux_arr, type_str, n_components, exposure_arr, wavelengths):
        sources, mixing, model = reduce_with_ica(flux_arr, n_components)
        np.savez(ica_data_file.format(type_str), sources=sources, mixing=mixing,
                    exposures=exposure_arr, wavelengths=wavelengths)
        pickle_FastICA(model, target_type=type_str)

    _ica_reduce_and_save(con_flux_arr, "continuum", ica_continuum_n, con_exposure_arr, wavelengths)
    _ica_reduce_and_save(flux_arr, "noncontinuum", ica_noncontinuum_n, exposure_arr, wavelengths)
    _ica_reduce_and_save(comb_flux_arr, "combined", ica_noncontinuum_n, exposure_arr, wavelengths)

    def _spca_reduce_and_save(flux_arr, type_str, n_components, exposure_arr, wavelengths):
        sources, components, model = reduce_with_spca(flux_arr, n_components)
        np.savez(spca_data_file.format(type_str), sources=sources, components=components,
                    exposures=exposure_arr, wavelengths=wavelengths)
        pickle_SPCA(model, target_type=type_str)

    _spca_reduce_and_save(con_flux_arr, "continuum", ica_continuum_n, con_exposure_arr, wavelengths)
    _spca_reduce_and_save(flux_arr, "noncontinuum", ica_noncontinuum_n, exposure_arr, wavelengths)
    _spca_reduce_and_save(comb_flux_arr, "combined", ica_noncontinuum_n, exposure_arr, wavelengths)


def pickle_FastICA(model, path='.', target_type='continuum', filename=None):
    if filename is None:
        filename = ica_pickle_file.format(target_type)
    output = open(os.path.join(path, filename), 'wb')
    pickle.dump(model, output)
    output.close()

def pickle_SPCA(model, path='.', target_type='continuum', filename=None):
    pickle_FastICA(model, path, target_type, spca_pickle_file.format(target_type) if filename is None else filename)

def unpickle_FastICA(path='.', target_type='continuum', filename=None):
    if filename is None:
        filename = ica_pickle_file.format(target_type)
    output = open(os.path.join(path, filename), 'rb')
    model = pickle.load(output)
    output.close()

    return model

def unpickle_SPCA(path='.', target_type='continuum', filename=None):
    return unpickle_FastICA(path, target_Type, spca_pickle_file.format(target_type) if filename is None else filename)


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

def get_SPCA(target_type='continuum', n=None):
    if n is not None:
        return SparsePCA(n_components = n, max_iter=spca_max_iter,
                        random_state=ica_random_state, n_jobs=-1)
    else:
        if target_type == 'continuum':
            return SparsePCA(n_components = ica_continuum_n, max_iter=spca_max_iter,
                        random_state=ica_random_state, n_jobs=-1)
        elif target_type == 'noncontinuum':
            return SparsePCA(n_components = ica_noncontinuum_n, max_iter=spca_max_iter,
                        random_state=ica_random_state, n_jobs=-1)
        elif target_type == 'combined':
            return SparsePCA(n_components = ica_noncontinuum_n, max_iter=spca_max_iter,
                        random_state=ica_random_state, n_jobs=-1)

    return None



def reduce_with_ica(flux_arr, n, mixing=None):
    ica = get_FastICA(n=n, mixing=mixing)
    if mixing is None:
        sources = ica.fit(flux_arr)

    sources = ica.transform(flux_arr, copy=True)

    return sources, ica.mixing_, ica

def reduce_with_spca(flux_arr, n):
    spca = get_SPCA(n=n)
    sources = spca.fit_transform(flux_arr) #

    return sources, spca.components_, spca


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

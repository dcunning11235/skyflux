import numpy as np
import matplotlib.pyplot as plt

from astropy.table import Table
from astropy.table import Column
from astropy.table import vstack
from astropy.table import join

from sklearn.decomposition import FastICA
from sklearn.decomposition import PCA
from sklearn.decomposition import SparsePCA
from sklearn import preprocessing as skpp

import fnmatch
import os
import os.path
import sys
import random
import pickle

ica_max_iter = 1000
spca_max_iter = 2500
pca_max_iter = 1000

ica_random_state = 1234975
ica_continuum_n = 40
ica_noncontinuum_n = 40

ica_data_file = "fastica_{}_{}_sources_and_mixing.npz"
spca_data_file = "spca_{}_{}_sources_and_components.npz"
pca_data_file = "pca_{}_{}_sources_and_components.npz"

ica_pickle_file = "fastica_{}_{}_pickle.pkl"
spca_pickle_file = "spca_{}_{}_pickle.pkl"
pca_pickle_file = "pca_{}_{}_pickle.pkl"

attempt_ica=True
attempt_spca=False #True
attempt_pca=False #True

def main():
    path = "."
    filter_split_path = None
    mixing_matrix_path = None
    filter_cutpoint = 20.0

    if len(sys.argv) == 2:
        path = sys.argv[1]
    if len(sys.argv) == 3:
        path = sys.argv[1]
        filter_split_path = sys.argv[2]
    if len(sys.argv) == 4:
        path = sys.argv[1]
        filter_split_path = sys.argv[2]
	filter_cutpoint = float(sys.argv[3])
    #if len(sys.argv) == 3:
    #    ica_continuum_n = int(sys.argv[2])
    #    ica_noncontinuum_n = ica_continuum_n

    '''
    con_flux_arr, con_exposure_arr, con_masks, con_wavelengths = load_all_in_dir(path,
                                                    use_con_flux=True, recombine_flux=False)
    flux_arr, exposure_arr, masks, wavelengths = load_all_in_dir(path, use_con_flux=False,
                                                    recombine_flux=False)
    comb_flux_arr, comb_exposure_arr, comb_masks, comb_wavelengths = \
        con_flux_arr + flux_arr, con_exposure_arr[:], con_masks[:], con_wavelengths[:]
    '''

    comb_flux_arr, comb_exposure_arr, comb_ivar_arr, comb_masks, comb_wavelengths = \
                load_all_in_dir(path, use_con_flux=False, recombine_flux=False,
                                pattern="stacked*exp??????.csv", ivar_cutoff=0.001)
    filter_split_arr = None
    if filter_split_path is not None:
        fstable = Table.read(filter_split_path, format="ascii.csv")
	filter_split_arr = fstable["flux_kurtosis_per_wl"] < filter_cutpoint

    print "Using a filter_split_path of:", filter_split_path, "and a cutpoint of:", filter_cutpoint
    print "filter_split_arr:", filter_split_arr, "with", np.sum(filter_split_arr), "True's"

    #mask_summed = np.sum(con_masks, axis=0)
    mask_summed = np.sum(comb_masks, axis=0)

    min_val_ind = np.min(np.where(mask_summed == 0))
    max_val_ind = np.max(np.where(mask_summed == 0))
    print min_val_ind, max_val_ind

    '''
    for i in range(con_flux_arr.shape[0]):
        con_flux_arr[i,:min_val_ind] = 0
        con_flux_arr[i,max_val_ind+1:] = 0
    '''

    def _ica_reduce_and_save(flux_arr, type_str, n_components, exposure_arr, wavelengths, filter_split_arr=None, which_filter=True):
        which_filter_str = "both"
        new_flux_arr = flux_arr
        if filter_split_arr is not None:
            new_flux_arr = np.array(flux_arr, copy=True)

            if which_filter:
                which_filter_str = "nonem"
                new_flux_arr[:,filter_split_arr] = 0
            else:
                which_filter_str = "em"
                new_flux_arr[:,~filter_split_arr] = 0
	
        sources, mixing, model = reduce_with_ica(new_flux_arr, n_components)
        np.savez(ica_data_file.format(type_str, which_filter_str), sources=sources, mixing=mixing,
                exposures=exposure_arr, wavelengths=wavelengths, filter_split_arr=filter_split_arr)
        pickle_FastICA(model, target_type=type_str, filter_str=which_filter_str)

    if attempt_ica:
        #_ica_reduce_and_save(con_flux_arr, "continuum", ica_continuum_n, con_exposure_arr, wavelengths)
        #_ica_reduce_and_save(flux_arr, "noncontinuum", ica_noncontinuum_n, exposure_arr, wavelengths)
        _ica_reduce_and_save(comb_flux_arr, "combined", ica_noncontinuum_n, comb_exposure_arr, comb_wavelengths)
        if filter_split_arr is not None:
            _ica_reduce_and_save(comb_flux_arr, "combined", ica_noncontinuum_n, comb_exposure_arr,
                                comb_wavelengths, filter_split_arr=filter_split_arr, which_filter=True)
            _ica_reduce_and_save(comb_flux_arr, "combined", ica_noncontinuum_n, comb_exposure_arr,
                                comb_wavelengths, filter_split_arr=filter_split_arr, which_filter=False)

    def _spca_reduce_and_save(flux_arr, type_str, n_components, exposure_arr, wavelengths):
        sources, components, model = reduce_with_spca(flux_arr, n_components)
        np.savez(spca_data_file.format(type_str), sources=sources, components=components,
                    exposures=exposure_arr, wavelengths=wavelengths)
        pickle_SPCA(model, target_type=type_str)

    if attempt_spca:
        #_spca_reduce_and_save(con_flux_arr, "continuum", ica_continuum_n, con_exposure_arr, wavelengths)
        #_spca_reduce_and_save(flux_arr, "noncontinuum", ica_noncontinuum_n, exposure_arr, wavelengths)
        _spca_reduce_and_save(comb_flux_arr, "combined", ica_noncontinuum_n, comb_exposure_arr, comb_wavelengths)

    def _pca_reduce_and_save(flux_arr, type_str, n_components, exposure_arr, wavelengths, filter_split_arr=None, which_filter=True):
        which_filter_str = "both"
        new_flux_arr = flux_arr
        if filter_split_arr is not None:
            new_flux_arr = np.array(flux_arr, copy=True)

            if which_filter:
                which_filter_str = "nonem"
                new_flux_arr[:,filter_split_arr] = 0
            else:
                which_filter_str = "em"
                new_flux_arr[:,~filter_split_arr] = 0

        sources, components, model = reduce_with_pca(new_flux_arr, n_components)
        np.savez(pca_data_file.format(type_str, which_filter_str), sources=sources, components=components,
                    exposures=exposure_arr, wavelengths=wavelengths, filter_split_arr=filter_split_arr)
        pickle_PCA(model, target_type=type_str, filter_str=which_filter_str)
	print model.explained_variance_ratio_, np.sum(model.explained_variance_ratio_)

    if attempt_pca:
        #_pca_reduce_and_save(con_flux_arr, "continuum", ica_continuum_n, con_exposure_arr, wavelengths)
        #_pca_reduce_and_save(flux_arr, "noncontinuum", ica_noncontinuum_n, exposure_arr, wavelengths)
        _pca_reduce_and_save(comb_flux_arr, "combined", ica_noncontinuum_n, comb_exposure_arr, comb_wavelengths)
        if filter_split_arr is not None:
            _pca_reduce_and_save(comb_flux_arr, "combined", ica_noncontinuum_n, comb_exposure_arr,
                                comb_wavelengths, filter_split_arr=filter_split_arr, which_filter=True)
            _pca_reduce_and_save(comb_flux_arr, "combined", ica_noncontinuum_n, comb_exposure_arr,
                                comb_wavelengths, filter_split_arr=filter_split_arr, which_filter=False)



def pickle_FastICA(model, path='.', target_type='continuum', filter_str='both', filename=None):
    if filename is None:
        filename = ica_pickle_file.format(target_type, filter_str)
    output = open(os.path.join(path, filename), 'wb')
    pickle.dump(model, output)
    output.close()

def pickle_SPCA(model, path='.', target_type='continuum', filter_str='both', filename=None):
    pickle_FastICA(model, path, target_type, filter_str, spca_pickle_file.format(target_type, filter_str) if filename is None else filename)

def pickle_PCA(model, path='.', target_type='continuum', filter_str='both', filename=None):
    pickle_FastICA(model, path, target_type, filter_str, pca_pickle_file.format(target_type, filter_str) if filename is None else filename)


def unpickle_FastICA(path='.', target_type='continuum', filter_str='both', filename=None):
    if filename is None:
        filename = ica_pickle_file.format(target_type, filter_str)
    output = open(os.path.join(path, filename), 'rb')
    model = pickle.load(output)
    output.close()

    return model

def unpickle_SPCA(path='.', target_type='continuum', filter_str='both', filename=None):
    return unpickle_FastICA(path, target_type, filter_str, spca_pickle_file.format(target_type, filter_str) if filename is None else filename)

def unpickle_PCA(path='.', target_type='continuum', filter_str='both', filename=None):
    return unpickle_FastICA(path, target_type, filter_str, pca_pickle_file.format(target_type, filter_str) if filename is None else filename)


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

def get_PCA(target_type='continuum', n=None):
    if n is not None:
        return PCA(n_components = n)
    else:
        if target_type == 'continuum':
            return PCA(n_components = ica_continuum_n)
        elif target_type == 'noncontinuum':
            return PCA(n_components = ica_noncontinuum_n)
        elif target_type == 'combined':
            return PCA(n_components = ica_noncontinuum_n)

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

def reduce_with_pca(flux_arr, n):
    pca = get_PCA(n=n)
    sources = pca.fit_transform(flux_arr) #

    return sources, pca.components_, pca


def load_all_in_dir(path, use_con_flux=True, recombine_flux=False, pattern = "stacked*-continuum.csv", ivar_cutoff=0):
    flux_list = []
    exp_list = []
    mask_list = []
    ivar_list = []
    wavelengths = None

    for file in os.listdir(path):
        if fnmatch.fnmatch(file, pattern):
            data = Table(Table.read(os.path.join(path, file), format="ascii.csv"), masked=True)
            mask = data['ivar'] <= ivar_cutoff
            ivar_list.append(np.array(data['ivar'], copy=False))

            exp = file.split("-")[2][3:]
            if exp.endswith("csv"):
                exp = int(exp[:-4])
            else:
                exp = int(exp)

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
    ivar_arr = np.array(ivar_list)

    return flux_arr, exp_arr, ivar_arr, mask_arr, wavelengths

if __name__ == '__main__':
    main()

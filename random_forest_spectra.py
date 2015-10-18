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

    results = None
    wavelengths = None
    for target_type in target_types:
        if test_inds[0] == 'ALL':
            test_inds = range(int(test_inds[1]), int(test_inds[2]))
        else:
            test_inds = [int(test_inds[0])]
        results = load_plot_etc_target_type(metadata_path, spectra_path, test_inds, target_type, no_plot=True, save_out=True, restrict_delta=True)
    '''
    for result in results:
        plt.plot(wavelengths, result)
        plt.plot(wavelengths, [0]*len(wavelengths))
        plt.legend(['Total Delta'])
        plt.tight_layout()
        plt.show()
        plt.close()
    '''

    print results[2]/len(test_inds)
    print results[3]/len(test_inds)-np.power(results[2]/len(test_inds),2)


def load_plot_etc_target_type(metadata_path, spectra_path, test_inds, target_type, no_plot=False, save_out=False, restrict_delta=False):
    obs_metadata = trim_observation_metadata(load_observation_metadata(metadata_path))
    c_sources, c_mixing, c_exposures, c_wavelengths = load_spectra_data(spectra_path, target_type=target_type)

    reduced_obs_metadata = obs_metadata[np.in1d(obs_metadata['EXP_ID'], c_exposures)]
    reduced_obs_metadata.sort('EXP_ID')
    sorted_inds = np.argsort(c_exposures)

    errs = np.empty((3,), dtype=float)

    rfr = ensemble.RandomForestRegressor(n_estimators=200, min_samples_split=1,
                        random_state=rfr_random_state, n_jobs=-1, verbose=False)
    knn = neighbors.KNeighborsRegressor(weights='distance', n_neighbors=10, p=64)

    reduced_obs_metadata.remove_column('EXP_ID')
    #reduced_obs_metadata.remove_column('AIRMASS')
    md_len = len(reduced_obs_metadata)
    X_arr = np.array(reduced_obs_metadata).view('f8').reshape((md_len,-1))

    ica = ICAize.unpickle_FastICA(path=spectra_path, target_type=target_type)


    ################################################################
    results = None
    for test_ind in test_inds:
        test_X = X_arr[test_ind]
        train_X = np.vstack( [X_arr[:test_ind], X_arr[test_ind+1:]] )
        test_y =  (c_sources[sorted_inds])[test_ind]
        train_y = np.vstack( [(c_sources[sorted_inds])[:test_ind], (c_sources[sorted_inds])[test_ind+1:]] )

        title_str = "exp{}, {}".format(c_exposures[sorted_inds[test_ind]], target_type)

        rfr.fit(X=train_X, y=train_y)
        knn.fit(X=train_X, y=train_y)

        rfr_prediction = rfr.predict(test_X)
        rfr_predicted_continuum = ica.inverse_transform(rfr_prediction, copy=True)

        print test_ind, c_exposures[sorted_inds[test_ind]],

        data = None
        actual = None
        mask = None
        delta_mask = None

        for file in os.listdir(spectra_path):
            if fnmatch.fnmatch(file, "stacked_sky_*exp{}-continuum.csv".format(c_exposures[sorted_inds[test_ind]])):
                data = Table.read(os.path.join(spectra_path, file), format="ascii.csv")
                mask = (data['ivar'] == 0) | np.isclose(rfr_predicted_continuum[0], 0)
                if restrict_delta:
                    delta_mask = mask.copy()
                    delta_mask[:2700] = True
                else:
                    delta_mask = mask

                actual = data['flux']
                if target_type == 'continuum':
                    actual = data['con_flux']
                elif target_type == 'combined':
                    actual += data['con_flux']

                rfr_delta = rfr_predicted_continuum[0] - actual
                if not no_plot:
                    plt.plot(c_wavelengths[~mask], rfr_predicted_continuum[0][~mask])
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
        knn_predicted_continuum = ica.inverse_transform(knn_prediction, copy=True)

        if not no_plot:
            plt.plot(c_wavelengths[~mask], knn_predicted_continuum[0][~mask])
            plt.plot(c_wavelengths[~mask], actual[~mask])
        knn_delta = knn_predicted_continuum[0] - actual
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

        avg_predicted_continuum = (knn_predicted_continuum + rfr_predicted_continuum)/2

        if not no_plot:
            plt.plot(c_wavelengths[~mask], avg_predicted_continuum[0][~mask])
            plt.plot(c_wavelengths[~mask], actual[~mask])
        avg_delta = avg_predicted_continuum[0] - actual
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
        print err_term, err_sum
        errs[2] = err_term

        if save_out:
            out_table = Table()
            wavelength_col = Column(c_wavelengths, name="wavelength", dtype=float)
            rf_col = Column(rfr_predicted_continuum[0], name="rf_flux", dtype=float)
            knn_col = Column(knn_predicted_continuum[0], name="knn_flux", dtype=float)
            avg_col = Column(avg_predicted_continuum[0], name="avg_flux", dtype=float)
            mask_col = Column(~mask, name="mask_col", dtype=bool)
            out_table.add_columns([wavelength_col, rf_col, knn_col, avg_col, mask_col])
            out_table.write("predicted_sky_exp{}.csv".format(c_exposures[sorted_inds[test_ind]]), format="ascii.csv")

        if results is None:
            results = [rfr_delta, knn_delta, errs, np.empty((3,), dtype=float)]
        else:
            results[0] += rfr_delta
            results[1] += knn_delta
            results[2] += errs
            results[3] += np.power(errs,2)

    return results


if __name__ == '__main__':
    main()

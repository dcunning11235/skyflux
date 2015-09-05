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
import gradient_boost_peaks as gbk

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

    con_flux_arr, con_exposure_list, con_wavelengths = load_all_in_dir(path,
                                                    use_con_flux=True, recombine_flux=False)
    flux_arr, exposure_list, wavelengths = load_all_in_dir(path, use_con_flux=False,
                                                    recombine_flux=False)

    print "Continuum Sky Flux"
    print "----------------------"
    con_perf_table = var_test_ica(con_flux_arr, con_exposure_list, con_wavelengths, low_n=10,
                hi_n=250, n_step=5, real_time_progress=True, idstr="continuum")
    con_perf_table.write("ica_performance_continuum.csv", format="ascii.csv")

    print "Non-continuum Sky Flux"
    print "----------------------"
    noncon_perf_table = var_test_ica(flux_arr, exposure_list, wavelengths, low_n=10, hi_n=250,
                n_step=5, real_time_progress=True, idstr="noncontinuum")
    noncon_perf_table.write("ica_performance_noncontinuum.csv", format="ascii.csv")

def var_test_ica(flux_arr_orig, exposure_list, wavelengths, low_n=3, hi_n=100, n_step=1, show_plots=False,
                    show_summary_plot=False, save_summary_plot=True, test_ind=7, real_time_progress=False,
                    idstr=None):

    start_ind = np.min(np.nonzero(flux_arr_orig[test_ind]))
    end_ind = np.max(np.nonzero(flux_arr_orig[test_ind]))

    perf_table = Table(names=["n", "avg_diff2", "max_diff_scaled"], dtype=["i4", "f4", "f4"])
    if hi_n > flux_arr_orig.shape[0]-1:
        hi_n = flux_arr_orig.shape[0]-1

    for n in range(low_n, hi_n, n_step):
        ica = FastICA(n_components = n, whiten=True, max_iter=750, random_state=1234975)
        test_arr = flux_arr_orig[test_ind].copy()

        flux_arr = np.vstack([flux_arr_orig[:test_ind], flux_arr_orig[test_ind+1:]])
        ica_flux_arr = flux_arr.copy()  #keep back one for testing
        ica.fit(ica_flux_arr)

        ica_trans = ica.transform(test_arr.copy(), copy=True)
        ica_rev = ica.inverse_transform(ica_trans.copy(), copy=True)

        avg_diff2 = np.ma.sum(np.ma.power(test_arr-ica_rev[0],2)) / (end_ind-start_ind)
        max_diff_scaled = np.ma.max(np.ma.abs(test_arr-ica_rev[0])) / (end_ind-start_ind)
        perf_table.add_row([n, avg_diff2, max_diff_scaled])

        if real_time_progress:
            print "n: {:4d}, avg (diff^2): {:0.5f}, scaled (max diff): {:0.5f}".format(n, avg_diff2, max_diff_scaled)

        if show_plots:
            plt.plot(wavelengths, test_arr)
            plt.plot(wavelengths, ica_rev[0])
            plt.plot(wavelengths, test_arr-ica_rev[0])

            plt.legend(['orig', 'ica', 'orig-ica'])
            plt.xlim((wavelengths[start_ind], wavelengths[end_ind]))

            plt.title("n={}, avg (diff^2)={}".format(n, avg_diff2))
            plt.tight_layout()
            plt.show()
            plt.close()

    if show_summary_plot or save_summary_plot:
        plt.plot(perf_table['n'], perf_table['avg_diff2'])
        plt.plot(perf_table['n'], perf_table['max_diff_scaled'])
        plt.title("performance")
        plt.tight_layout()
        if show_summary_plot:
            plt.show()
        if save_summary_plot:
            if idstr is None:
                idstr = random.randint(1000000, 9999999)
            plt.savefig("ica_performance_{}.png".format(idstr))
        plt.close()

    return perf_table

if __name__ == '__main__':
    main()

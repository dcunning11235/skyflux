import numpy as np
import matplotlib.pyplot as plt

from astropy.table import Table
from astropy.table import Column
from astropy.table import vstack
from astropy.table import join

import fnmatch
import os
import os.path
import sys

def main():
    predicted_path = "."

    for file in os.listdir(predicted_path):
        if fnmatch.fnmatch(file, "predicted_sky*exp??????.csv"):
            predicted_data = Table.read(os.path.join(predicted_path, file), format="ascii.csv")
            exp_id = file.split("_")[-1][3:9]
            spectra_file = "stacked_sky_*exp{}-continuum.csv".format(exp_id)
            actual_data = None
            for data_file in os.listdir(predicted_path):
                if fnmatch.fnmatch(data_file, spectra_file):
                    actual_data = Table.read(os.path.join(predicted_path, data_file), format="ascii.csv")

            if actual_data is not None:
                actual_flux = actual_data['con_flux'] + actual_data['flux']

                rf_delta = predicted_data['rf_flux'] - actual_flux
                knn_delta = predicted_data['knn_flux'] - actual_flux
                avg_delta = predicted_data['avg_flux'] - actual_flux

                delta_mask = (predicted_data['mask_col'] == 'False')

                rf_err_term = np.sum(np.power(rf_delta[~delta_mask], 2))/len(rf_delta[~delta_mask])
                knn_err_term = np.sum(np.power(knn_delta[~delta_mask], 2))/len(knn_delta[~delta_mask])
                avg_err_term = np.sum(np.power(avg_delta[~delta_mask], 2))/len(avg_delta[~delta_mask])

                print exp_id, rf_err_term, knn_err_term, avg_err_term

if __name__ == '__main__':
    main()

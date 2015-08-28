import numpy as np

from astropy.table import Table
from astropy.table import Column
from astropy.table import vstack
from astropy.table import join

from sklearn import ensemble

import fnmatch
import os
import os.path
import sys

def load_all_in_dir(path):
    pattern = "stacked*-continuum-peaks.csv"
    tables_list = []
    exp_list = []
    lines_list = None

    for file in os.listdir(path):
        if fnmatch.fnmatch(file, pattern):
            data = Table.read(os.path.join(path, file), format="ascii.csv")

            exp = int(file.split("-")[2][3:])
            exp_col = Column([exp]*len(data), name="EXP_ID")
            data.add_column(exp_col)

            tables_list.append(data)
            exp_list.append(exp)

            if lines_list is None:
                lines_list = Table()
                lines_list.add_column(data['source'])
                lines_list.add_column(data['wavelength_target'])

    return vstack(tables_list), exp_list, lines_list

def load_observation_metadata(path, file = "annotated_metadata.csv"):
    data = Table.read(os.path.join(path, file), format="ascii.csv")

    return data

def trim_observation_metadata(data, copy=False):
    if copy:
        data = data.copy()

    kept_columns = ['EXP_ID', 'RA', 'DEC', 'AZ', 'ALT', 'AIRMASS',
                    'LUNAR_MAGNITUDE', 'LUNAR_ELV', 'LUNAR_SEP', 'SOLAR_ELV',
                    'SOLAR_SEP', 'GALACTIC_CORE_SEP', 'GALACTIC_PLANE_SEP']
                    #'EPHEM_DATE',
    removed_columns = [name for name in data.colnames if name not in kept_columns]
    data.remove_columns(removed_columns)

    return data

def get_X_and_y(data_subset, obs_metadata, use_con_flux=False):
    # X: n_samples, n_features
    # y: n_samples
    # E.g., for this case, X will be the observation metadata for each exposure and y will be
    #       the total flux for a given line/peak (actually x2:  total_flux and total_con_flux)
    y_col = 'total_flux' if not use_con_flux else 'total_con_flux'
    full_table = join(obs_metadata, data_subset['EXP_ID', y_col])

    labels = full_table['EXP_ID']
    full_table.remove_column('EXP_ID')
    y = full_table[y_col]
    full_table.remove_column(y_col)

    return full_table, y, labels

def ndarrayidze(table):
    temp_arr = np.array(table, copy=False)
    length = len(temp_arr)
    width = 0
    if length > 0:
        width = len(temp_arr[0])
    return temp_arr.view('<f8').reshape( (length, width) )

def get_feature_importances(data_table, obs_metadata, lines_table, use_con_flux=False):
    feature_importances_list = []
    X_colnames = None
    for line_name, line_wavelength in lines_table['source', 'wavelength_target']:
        subset = data_table[(data_table['source'] == line_name) & (data_table['wavelength_target'] == line_wavelength)]
        X, y, labels = get_X_and_y(subset, obs_metadata, use_con_flux)
        if X_colnames is None:
            X_colnames = X.colnames

        params = {'n_estimators': 500, 'max_depth': 4, 'min_samples_split': 1,
                'learning_rate': 0.01, 'loss': 'lad'}
        clf = ensemble.GradientBoostingRegressor(**params)
        X = ndarrayidze(X)
        clf.fit(X, y)
        feature_importances_list.append(clf.feature_importances_)

    fi = np.array(feature_importances_list)
    fi_table = Table(fi, names = X_colnames)
    fi_table.add_column(lines_table['source'])
    fi_table.add_column(lines_table['wavelength_target'])

    return fi_table

def main():
    path = "."
    if len(sys.argv) == 2:
        path = sys.argv[1]
    data_table, exposure_list, lines_table = load_all_in_dir(path)
    obs_metadata = trim_observation_metadata(load_observation_metadata(path))

    fi_flux_table = get_feature_importances(data_table, obs_metadata, lines_table)
    fi_flux_table.write("gradient_boost_flux_fi.csv", format="ascii.csv")

    fi_con_flux_table = get_feature_importances(data_table, obs_metadata, lines_table, use_con_flux=True)
    fi_con_flux_table.write("gradient_boost_con_flux_fi.csv", format="ascii.csv")

if __name__ == '__main__':
    main()

import numpy as np

from astropy.table import Table
import fnmatch
import sys
import datetime as dt
import posixpath
import os

path='.'

def main():
    pattern = "stacked*exp??????.csv"
    flux_list = []
    exp_list = []
    mask_list = []
    wavelengths = None

    for file in os.listdir(path):
        if fnmatch.fnmatch(file, pattern):
            data = Table.read(os.path.join(path, file), format="ascii.csv")
            mask = data['ivar'] == 0

            exp = int(file.split("-")[2][3:9])

            #neg_mask = data['flux'][~mask] < -0.5
            neg_mask = data['flux'] < -0.5
            set_neg_mask = neg_mask & ~mask
            if np.any(set_neg_mask):
                print file, exp, data['wavelength'][set_neg_mask][0], data['flux'][set_neg_mask][0], "repairing..."
                data['ivar'][set_neg_mask] = 0

                data.write(os.path.join(path, file), format="ascii.csv")

if __name__ == '__main__':
    main()

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
            data = Table(Table.read(os.path.join(path, file), format="ascii.csv"), masked=True)
            mask = data['ivar'] == 0

            exp = int(file.split("-")[2][3:9])

            neg_mask = data['flux'][~mask] < -20
            if np.any(neg_mask):
                print file, exp, data['wavelength'][~mask][neg_mask][0], data['flux'][~mask][neg_mask][0]

if __name__ == '__main__':
    main()

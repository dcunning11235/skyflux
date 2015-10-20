import numpy as np

from astropy.table import Table

import sys
import datetime as dt
import posixpath

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

            exp = int(file.split("-")[2][3:])

            neg_mask = (data['con_flux'] + data['flux'])[~mask] < 20
            if np.any(mask):
                print "Found a file with extreme negative (<20):"
                print "\t", data['wavelength'][~mask][neg_mask]
                print "\t", data['con_flux'] + data['flux'])[~mask][neg_mask]

if __name__ == '__main__':
    main()

import numpy as np

from astropy.table import Table
from astropy.table import Column

import fnmatch
import os
import os.path
import sys

def main():
    path = "."
    pattern = ""
    if len(sys.argv) == 3:
        path = sys.argv[1]
        pattern = sys.argv[2]
    else:
        pattern = sys.argv[1]

    for file in os.listdir(path):
        if fnmatch.fnmatch(file, pattern):
            data = Table(Table.read(os.path.join(path, file), format="ascii"), masked=True)
            data.mask = [(data['ivar'] == 0)]*len(data.columns)

            if np.count_nonzero(data['con_flux'].data < 0):
                print file, "BAD:  continuum has zub-zero con_flux!!!!"
                print data['con_flux'][data['con_flux'].data < 0]

            total = data['con_flux']+data['flux']
            if np.count_nonzero(total < 0):
                print file, "WORSE:  total has zub-zero con_flux!!!!"
                print total[total < 0]

            ivar_cutoff = 0.005
            ivar_cutoff_mask = (data['ivar'].data < ivar_cutoff) & (data['ivar'].data > 0)
            if np.any(ivar_cutoff_mask):
                print file, "QUESTIONABLE:  ivar less than", ivar_cutoff
                print data["ivar"][ivar_cutoff_mask]

if __name__ == '__main__':
    main()

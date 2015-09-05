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
                print file, "has zub-zero con_flux!!!!"
                print data['con_flux'][data['con_flux'].data < 0]

if __name__ == '__main__':
    main()

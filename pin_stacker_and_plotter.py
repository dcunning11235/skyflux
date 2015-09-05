import numpy as np

from astropy.table import Table
from astropy.table import vstack

import fnmatch
import os
import os.path
import time
import sys
import datetime as dt
import posixpath

import bossdata.path as bdpath
import bossdata.remote as bdremote
import bossdata.spec as bdspec
import bossdata.plate as bdplate
import bossdata.bits as bdbits

from speclite import accumulate
from speclite import resample

import stack

import matplotlib.pyplot as plt

def main():
    path = "."
    pattern = ""
    if len(sys.argv) == 3:
        path = sys.argv[1]
        pattern = sys.argv[2]
    else:
        pattern = sys.argv[1]

    growth = size = 100
    pile_of_tables = [None]*size
    count = 0

    for file in os.listdir(path):
        if fnmatch.fnmatch(file, pattern):
            data = Table(Table.read(os.path.join(path, file), format="ascii"), masked=True)
            pile_of_tables[count] = data
            count += 1
            if count >= size:
                pile_of_tables.extend([None]*growth)
                size += growth
    all_data = vstack(pile_of_tables[:count])
    spec_1 = all_data[all_data['fiber'] <= 500]
    spec_2 = all_data[all_data['fiber'] > 500]
    cols = [name for name in all_data.colnames if name.startswith("target")]
    for col in cols:
        #avg = np.mean(all_data[col])
        #std = np.std(all_data[col])
        min_val = np.min(all_data[col])
        max_val = np.max(all_data[col])

        #print all_data[col]
        #plt.hist(all_data[col], bins=(max_val - min_val)/stack.skyexp_wlen_delta + 1)
        plt.hist([spec_1[col], spec_2[col]], bins=(max_val- min_val)/stack.skyexp_wlen_delta + 1)

        #bins = np.arange(min_val, max_val+stack.skyexp_wlen_delta, stack.skyexp_wlen_delta)
        #print np.histogram(all_data[col], bins)

        plt.tight_layout()
        plt.show()
        plt.close()

if __name__ == '__main__':
    main()

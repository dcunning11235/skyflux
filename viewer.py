import numpy as np
from astropy.table import Table
import sys
import matplotlib.pyplot as plt
import speclite.accumulate as accumulate
import fnmatch
import os
import os.path

def plot_it(data):
    data = Table(data, masked=True)
    data.mask = [data['ivar'] == 0]*len(data.columns)

    val = data['flux']+data['sky']
    sigma = np.power(data['ivar'], -0.5)

    plt.plot(data['wavelength'], val)
    plt.fill_between(data['wavelength'], val-sigma, val+sigma, color='red')
    plt.show()


data = None
result = None
path = "."
pattern = ""

if len(sys.argv) == 3:
    path = sys.argv[1]
    pattern = sys.argv[2]
else:
    pattern = sys.argv[1]

for file in os.listdir(path):
    if fnmatch.fnmatch(file, pattern):
        data2 = Table.read(os.path.join(path, file), format="ascii")
        if data is None:
            data = data2
        else:
            result = accumulate(np.array(data), np.array(data2), result, join="wavelength", add=('flux', 'sky'), weight='ivar')

if result is None:
    result = data
plot_it(result)

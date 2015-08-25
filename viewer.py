import numpy as np
from astropy.table import Table
import sys
import matplotlib.pyplot as plt
import speclite.accumulate as accumulate
import speclite.resample as resample
import fnmatch
import os
import os.path
import bossdata.spec as bdspec
from numpy.lib import recfunctions as rfn

def plot_it(data, key):
    data = Table(data, masked=True)
    if 'ivar' in data.colnames:
        data.mask = [(data['flux'] == 0) | (np.abs(data['ivar']) <= 0.0001)]*len(data.columns)
    else:
        data.mask = [(data['flux'] == 0)]*len(data.columns)

    val = data['flux']
    if 'sky' in data.colnames:
        val += data['sky']
    if 'ivar' in data.colnames:
        sigma = np.power(data['ivar'], -0.5)
    else:
        sigma = 0


    if 'con_flux' in data.colnames:
        plt.plot(data[key], [0]*len(data[key]), color='red')
        plt.plot(data[key], data['con_flux'], color='green')
        plt.plot(data[key], val + data['con_flux'], color='orange', alpha=0.7)
    else:
        plt.plot(data[key], val)
    if np.any(sigma > 0) and 'con_flux' not in data.colnames:
        plt.fill_between(data[key], val-sigma, val+sigma, color='red')
    plt.tight_layout()
    #plt.ylim((np.percentile(val-sigma,0.1),np.percentile(val+sigma,99.9)))
    plt.show()
    plt.close()

data = None
result = None
path = "."
pattern = ""
key = "wavelength"

if len(sys.argv) == 3:
    path = sys.argv[1]
    pattern = sys.argv[2]
else:
    pattern = sys.argv[1]

data_as_is = []
for file in os.listdir(path):
    if fnmatch.fnmatch(file, pattern):
        data2 = Table.read(os.path.join(path, file), format="ascii")
        #plot_it(data2, key)
        data_as_is.append(data2)

        if data is None:
            data = data2
            if 'wavelength' in data.colnames:
                key = 'wavelength'
            else:
                key = 'loglam'
        else:
            weight = 'ivar' if 'ivar' in data2.colnames else None
            add_names = [name for name in data2.colnames if name not in [weight, key]]
            result = accumulate(np.array(data), np.array(data2), result, join=key, add=add_names, weight=weight)

if result is None:
    result = data
plot_it(result, key)

'''
if 'loglam' not in result.dtype.names:
    x_out = np.arange(np.log10(3500.26), np.log10(3500.26) + 0.48 + 0.0001, 0.0001)
    x_out = 10**x_out
    new_data = resample(result.as_array(), 'wavelength', x_out, ('flux', 'sky', 'ivar'))
else:
    x_out = np.arange(np.log10(3500.26), np.log10(3500.26) + 0.48 + 0.0001, 0.0001)
    new_data = resample(result.as_array(), 'loglam', x_out, ('flux', 'sky', 'ivar'))

plot_it(new_data, key)
'''

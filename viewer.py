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

def plot_it(data, key, peaks_table=None, no_con_flux=False, unmask=False, data_col=None, title=None, stack=False):
    data = Table(data, masked=True)
    if data_col is None:
        data_col = 'flux'
    if not unmask:
        if 'ivar' in data.colnames:
            data.mask = [(data['ivar'] == 0) | (np.abs(data['ivar']) <= 0.0001)]*len(data.columns)
        else:
            data.mask = [(data[data_col] == 0)]*len(data.columns)

    val = data[data_col]
    if 'sky' in data.colnames:
        val += data['sky']
    if 'ivar' in data.colnames:
        sigma = np.power(data['ivar'], -0.5)
    else:
        sigma = 0


    if 'con_flux' in data.colnames and not no_con_flux:
        val += data['con_flux']

        plt.plot(data[key], [0]*len(data[key]), color='red')
        plt.plot(data[key], data['con_flux'], color='green')
        plt.plot(data[key], val, color='orange', alpha=0.7)
    else:
        if stack:
            plt.plot(data[key], val, alpha=0.7)
        else:
            plt.plot(data[key], [0]*len(data[key]), color='red')
            plt.plot(data[key], val, color='orange', alpha=0.7)

    if not stack and np.any(sigma > 0) and 'con_flux' not in data.colnames:
        plt.fill_between(data[key], val-sigma, val+sigma, color='red')

    if peaks_table is not None:
        for row in peaks_table:
            if row['total_type'] == 'line' and row['wavelength_lower_bound'] != 0:
                plt.fill_between([row['wavelength_lower_bound'], row['wavelength_upper_bound']],
                                [0, np.max(val)], color='gray')
    if title is not None:
        plt.suptitle(title)
    plt.tight_layout()
    #plt.ylim((np.percentile(val-sigma,0.1),np.percentile(val+sigma,99.9)))
    if not stack:
        plt.show()
        plt.close()

data = None
result = None
path = "."
pattern = ""
peaks = None
key = "wavelength"
data_col = None

if len(sys.argv) == 4:
    path = sys.argv[1]
    pattern = sys.argv[2]
    peaks = sys.argv[3]
elif len(sys.argv) == 3:
    #path = sys.argv[1]
    pattern = sys.argv[1]
    data_col = sys.argv[2]
else:
    pattern = sys.argv[1]

data_as_is = []
for file in os.listdir(path):
    if fnmatch.fnmatch(file, pattern):
        data2 = Table.read(os.path.join(path, file), format="ascii")
        plot_it(data2, key, data_col=data_col, title=file, stack=True)
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

plt.show()
plt.close()

if result is None:
    result = data

peaks_table = None
if peaks is not None:
    peaks_table = Table.read(os.path.join(path, peaks), format="ascii")

    plot_it(result, key, peaks_table, no_con_flux=False, unmask=True)

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

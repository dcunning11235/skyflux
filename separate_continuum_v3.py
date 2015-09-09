import numpy as np
from astropy.table import Table
from astropy.convolution import MexicanHat1DKernel
from astropy.convolution import Gaussian1DKernel
from astropy.convolution import convolve

from scipy.signal import general_gaussian
from scipy.signal import gaussian
from scipy.signal import fftconvolve
from scipy.signal import argrelextrema
from scipy.interpolate import InterpolatedUnivariateSpline
from scipy.interpolate import interp1d

import fnmatch
import os
import os.path
import sys
import time

import matplotlib.pyplot as plt

import stack
import measure_peaks

# Magic numbers.... oooohhhhhhhh!
block_sizes = np.array([1,2])
base_stds = np.array([0.5,0.5])
noisy_cutoffs = np.array([3.5])
noisy_sizes = np.array([6])
noisy_mult = np.array([15])
split_noisy_app = 7080 #2650 # = 6084.01 A

all_timing = False
main_timing = False
ts = time.time()

def main():
    path = "."
    pattern = ""
    if len(sys.argv) == 3:
        path = sys.argv[1]
        pattern = sys.argv[2]
    else:
        pattern = sys.argv[1]

    global ts
    ts = mark_time("start loop", ts)
    for file in os.listdir(path):
        if fnmatch.fnmatch(file, pattern):
            data = Table(Table.read(os.path.join(path, file), format="ascii"), masked=True)
            orig_mask = (data['ivar'] == 0)
            data.mask = [(data['ivar'] == 0)]*len(data.columns)
            idstr = file[:file.rfind('.')]

            test_data = data.copy()

            test_data['flux'] = minimize(test_data['wavelength'], test_data['flux'], 100, 0, start_ind=2650)

            window_size = 151
            sigma=24
            #window = general_gaussian(window_size, p=0.5, sig=sigma, sym=True)
            window = gaussian(window_size, std=sigma, sym=True) +  gaussian(window_size, std=sigma*4, sym=True)
            filtered = test_data['flux'].copy()
            wavelength = test_data['wavelength'].data
            mask = filtered.mask.copy()
            if np.any(filtered.mask) and np.any(~filtered.mask):
                filtered[mask] = np.interp(wavelength[mask], wavelength[~mask], filtered[~mask])
            filtered = fftconvolve(window, filtered)
            filtered = (np.ma.average(test_data['flux']) / np.ma.average(filtered)) * filtered
            print test_data['wavelength'].shape, filtered.shape
            filtered = np.roll(filtered, -(window_size-1)/2)[:-(window_size-1)]

            plt.plot(test_data['wavelength'], test_data['flux'])
            print test_data['wavelength'].shape, filtered.shape
            plt.plot(test_data['wavelength'], filtered)
            plt.title("Gaussian Convolve 151, 24, --")
            plt.tight_layout()
            plt.show()
            plt.close()

            test_data['flux'] = np.ma.min(np.ma.vstack([test_data['flux'], filtered]), axis=0)
            plt.plot(test_data['wavelength'], test_data['flux'])
            plt.title("Min'ed w/ Gaussian Convolve 151, 24, --")
            plt.tight_layout()
            plt.show()
            plt.close()



            window_size = 121
            sigma=18
            window = gaussian(window_size, std=sigma, sym=True) +  gaussian(window_size, std=sigma*4, sym=True)
            filtered = test_data['flux'].copy()
            wavelength = test_data['wavelength'].data
            mask = filtered.mask.copy()
            if np.any(filtered.mask) and np.any(~filtered.mask):
                filtered[mask] = np.interp(wavelength[mask], wavelength[~mask], filtered[~mask])
            filtered = fftconvolve(window, filtered)
            filtered = (np.ma.average(test_data['flux']) / np.ma.average(filtered)) * filtered
            print test_data['wavelength'].shape, filtered.shape
            filtered = np.roll(filtered, -(window_size-1)/2)[:-(window_size-1)]

            plt.plot(test_data['wavelength'], test_data['flux'])
            print test_data['wavelength'].shape, filtered.shape
            plt.plot(test_data['wavelength'], filtered)
            plt.title("Gaussian Convolve 121, 24, --")
            plt.tight_layout()
            plt.show()
            plt.close()

            test_data['flux'] = np.ma.min(np.ma.vstack([test_data['flux'], filtered]), axis=0)
            plt.plot(test_data['wavelength'], test_data['flux'])
            plt.title("Min'ed w/ Gaussian Convolve 121, 24, --")
            plt.tight_layout()
            plt.show()
            plt.close()



            continuum = split_spectrum(test_data['wavelength'], test_data['flux'])
            wo_continuum = data['flux'] - continuum

            #continuum, wo_continuum = tamp_down(data['wavelength'], continuum, wo_continuum, span=51)
            #continuum, wo_continuum = tamp_down(data['wavelength'], continuum.data, wo_continuum.data)
            #continuum, wo_continuum = tamp_down(data['wavelength'], continuum, wo_continuum, span=31)
            #continuum, wo_continuum = tamp_down(data['wavelength'], continuum, wo_continuum, span=21)
            #continuum, wo_continuum = tamp_down(data['wavelength'], continuum, wo_continuum, span=11)

            #continuum, wo_continuum = smooth(continuum, wo_continuum)

            save_data(data['wavelength'], wo_continuum, continuum, data['ivar'], orig_mask, idstr)
            ts = mark_time("save_data", ts)

def tamp_down(wavelength, continuum, wo_continuum, span=41):
    work_continuum = np.ma.array(continuum)
    work_wo_continuum = np.ma.array(wo_continuum)
    total = work_wo_continuum + work_continuum

    chunked_continuum, begin_orig_mask, end_orig_mask = trim_array_from_mask(work_continuum, work_continuum.mask, buffer=5)
    chunked_continuum, block_diff, block_remainder = chunk_array(chunked_continuum, begin_orig_mask,
                                                        end_orig_mask, span*3)
    chunked_continuum[:] = np.ma.mean(chunked_continuum, axis=1)[:, np.newaxis]
    chunked_continuum = chunked_continuum.reshape((chunked_continuum.size, ) )

    averages = np.zeros(continuum.size, dtype=float)
    averages[begin_orig_mask:end_orig_mask+1] = chunked_continuum[:-block_diff] if block_remainder > 0 else chunked_continuum

    def _moving_average(a, n=3):
        ret = np.cumsum(a, dtype=float)
        ret[n:] = ret[n:] - ret[:-n]
        ret = np.ma.concatenate([a[:(n-1)/2], ret[n-1:]/n, a[-(n-1)/2:]])
        return ret

    move_avg_cont = _moving_average(work_continuum, n=span)
    too_loo_mask = move_avg_cont < averages

    move_avg_cont -= work_continuum
    move_avg_cont[move_avg_cont > 0] = 0
    move_avg_cont[too_loo_mask] = 0

    work_continuum += move_avg_cont
    work_wo_continuum = total - work_continuum

    plt.plot(wavelength, continuum)
    plt.title("Tamping with span={}".format(span))
    plt.tight_layout()
    plt.show()
    plt.close()

    return work_continuum, work_wo_continuum

def smooth(continuum, wo_continuum):
    total = wo_continuum + continuum

    g = Gaussian1DKernel(stddev=2)
    continuum = convolve(continuum, g, boundary='extend')

    wo_continuum = total - continuum

    return continuum, wo_continuum

def save_data(wlen, flux, con_flux, ivar, mask, idstr):
    wlen.mask = np.ma.nomask
    flux.mask = mask
    con_flux = np.ma.array(con_flux, mask=mask)
    #con_flux.mask = mask
    ivar.mask = mask
    continuum_table = Table([wlen.data, flux.filled(0), con_flux.filled(0), ivar.filled(0)], names=["wavelength", "flux", "con_flux", "ivar"])
    continuum_table.write("{}-continuum.csv".format(idstr), format="ascii.csv")

def minimize(work_wlen, work_data, block_size, noise_cutoff, start_ind=0):
    working_mask = work_data.mask.copy()
    working_mask[:start_ind] = True
    unmasked_inds = np.where(working_mask == False)
    first_ind = np.min(unmasked_inds)
    last_ind = np.max(unmasked_inds)
    working_block = work_data[first_ind:last_ind+1]

    work_len = last_ind + 1 - first_ind
    len_overage = work_len % block_size
    extent_size = 0
    if len_overage > 0:
        extent_size = block_size - len_overage
        working_block = np.ma.concatenate((working_block, working_block[-extent_size:]))

    working_block = working_block.reshape((-1,block_size))
    stds = None
    if noise_cutoff > 0:
        stds = np.ma.std(working_block, axis=1)
    mins = np.ma.min(working_block.data, axis=1)
    masks = np.ma.count_masked(working_block, axis=1)

    mins[masks > block_size/5] = 0
    if noise_cutoff > 0:
        stds[masks > block_size/5] = 0

    if noise_cutoff > 0:
        min_mask = (stds > noise_cutoff) | (stds == 0)
    else:
        min_mask = np.ones((mins.size,), dtype=bool)
    working_block[min_mask, :] =  mins[min_mask,np.newaxis]
    working_block = working_block.reshape((working_block.size,))
    if extent_size > 0:
        working_block = working_block[:-extent_size]

    temp_block = work_data.copy()
    temp_block[first_ind:last_ind+1] = working_block[:]
    working_block = temp_block
    print work_wlen.shape, working_block.shape

    plt.plot(work_wlen, working_block)
    plt.title("Minimized with span={}".format(block_size))
    plt.tight_layout()
    plt.show()
    plt.close()

    return working_block


def split_spectrum(work_wlen, work_data):
    work_wlen_cp = work_wlen.copy()
    work_data_cp = work_data.copy()

    mask = work_data.mask.copy()

    window_size = 61
    sigma=12
    #window = general_gaussian(window_size, p=0.5, sig=sigma, sym=True)
    window = gaussian(window_size, std=sigma, sym=True)
    filtered = work_data_cp.copy()
    if np.any(filtered.mask) and np.any(~filtered.mask):
        filtered[mask] = np.interp(work_wlen_cp[mask], work_wlen_cp[~mask], work_data_cp[~mask])
    if np.any(work_data_cp.mask) and np.any(~work_data_cp.mask):
        work_data_cp[mask] = np.interp(work_wlen_cp[mask], work_wlen_cp[~mask], work_data_cp[~mask])
    filtered = fftconvolve(window, filtered)
    print "masked count:", np.ma.count_masked(work_data_cp), np.ma.count_masked(filtered)
    filtered = (np.ma.average(work_data_cp) / np.ma.average(filtered)) * filtered
    filtered = np.roll(filtered, -(window_size-1)/2)[:-(window_size-1)]

    plt.plot(work_wlen, work_data_cp)
    plt.plot(work_wlen, filtered)
    plt.title("Gaussian Convolve 41, 4, 1")
    plt.tight_layout()
    plt.show()
    plt.close()

    window = general_gaussian(window_size, p=0.5, sig=sigma, sym=True)
    #window = gaussian(window_size, std=sigma, sym=True)
    filtered = work_data_cp.copy()
    if np.any(filtered.mask) and np.any(~filtered.mask):
        filtered[mask] = np.interp(work_wlen_cp[mask], work_wlen_cp[~mask], work_data_cp[~mask])
    if np.any(work_data_cp.mask) and np.any(~work_data_cp.mask):
        work_data_cp[mask] = np.interp(work_wlen_cp[mask], work_wlen_cp[~mask], work_data_cp[~mask])
    filtered = fftconvolve(window, filtered)
    print "masked count:", np.ma.count_masked(work_data_cp), np.ma.count_masked(filtered)
    filtered = (np.ma.average(work_data_cp) / np.ma.average(filtered)) * filtered
    filtered = np.roll(filtered, -(window_size-1)/2)[:-(window_size-1)]

    plt.plot(work_wlen, work_data_cp)
    plt.plot(work_wlen, filtered)
    plt.title("General Gaussian Convolve 41, 4, 1")
    plt.tight_layout()
    plt.show()
    plt.close()

    work_data_cp = np.ma.min(np.vstack([filtered, work_data_cp]), axis=0)

    plt.plot(work_wlen, work_data_cp)
    plt.title("Min on convolve and spectrum")
    plt.tight_layout()
    plt.show()
    plt.close()

    cont_flux = work_data_cp
    return cont_flux

def chunk_array(data, begin_orig_mask, end_orig_mask, block_size, extend_func=np.ma.mean, extent_val=None, extent_type=float):
    data_len = end_orig_mask - begin_orig_mask + 1
    block_remainder = data_len % block_size
    block_diff = (block_size - block_remainder) % block_size

    #new_shape is shape, plus one extra row for partal data
    new_shape = ( (data_len-block_remainder)/block_size + (1 if block_remainder > 0 else 0), block_size )

    if block_remainder > 0:
        #For now, just extend data... play with this later
        if extend_func is not None:
            extent_val = extend_func(data[-block_diff:])
        extent = np.full( block_diff, extent_val, dtype=extent_type )
        data = np.ma.concatenate([data, extent])
    data = np.array(data).reshape(new_shape)

    return data, block_diff, block_remainder

def trim_array_from_mask(data, orig_mask, buffer=0):
    orig_mask_extents = np.where(~orig_mask)
    begin_orig_mask = np.min(orig_mask_extents)-buffer
    end_orig_mask = np.max(orig_mask_extents)+buffer
    if begin_orig_mask < 0:
        begin_orig_mask = 0
    if end_orig_mask >= orig_mask.size:
        end_orig_mask = orig_mask.size-1

    return data[begin_orig_mask:end_orig_mask+1], begin_orig_mask, end_orig_mask

def kill_peaks(work_wlen_cp, work_data_cp, peaks_mask, orig_mask, block_size, cutoff=None,
                is_noisy=False):
    combined_mask = peaks_mask | orig_mask

    work_wlen_cut, begin_orig_mask, end_orig_mask = trim_array_from_mask(work_wlen_cp, orig_mask)
    work_data_cut = np.array(work_data_cp[begin_orig_mask:end_orig_mask+1])
    combined_mask = combined_mask[begin_orig_mask:end_orig_mask+1]

    '''
    Need to only consider values inside the orig mask:  We will set the head and tail
    afterward to the average of the (un-orig-masked, un-peak-masked) values around
    the two ends.
    '''
    work_data_cut[combined_mask] = np.interp(work_wlen_cut[combined_mask], work_wlen_cut[~combined_mask], work_data_cut[~combined_mask])

    if not is_noisy:
        work_data_cp[begin_orig_mask:end_orig_mask+1] = work_data_cut
        return work_data_cp

    work_data_cut, block_diff, block_remainder = chunk_array(work_data_cut, begin_orig_mask,
                                                    end_orig_mask, block_size)
    combined_mask, block_diff, block_remainder = chunk_array(combined_mask, begin_orig_mask,
                                                    end_orig_mask, block_size, extend_func=None,
                                                    extent_val=False, extent_type=bool)

    masked_count = np.ma.sum(combined_mask, axis=1)
    stdevs = np.ma.std(work_data_cut, axis=1)
    overs = np.ma.min(work_data_cut, axis=1)
    ins = np.ma.mean(work_data_cut, axis=1)

    #stdevs[masked_count > (block_size * 0.90)] = 0
    x = np.arange(len(stdevs))
    mask = (stdevs == 0) & (overs == 0)

    stdevs[mask] = np.interp(x[mask], x[~mask], stdevs[~mask])
    overs[mask] = np.interp(x[mask], x[~mask], overs[~mask])
    ins[mask] = np.interp(x[mask], x[~mask], ins[~mask])

    if not is_noisy:
        stdev_rows = np.where(stdevs <= cutoff)
        work_data_cut[stdev_rows,:] = ins[stdev_rows,np.newaxis]
        stdev_rows = np.where(stdevs > cutoff)
        work_data_cut[stdev_rows,:] = overs[stdev_rows,np.newaxis]
    else:
        stdev_rows = np.where((stdevs > cutoff) | (masked_count > (block_size/3)))
        work_data_cut[stdev_rows,:] = overs[stdev_rows,np.newaxis]

    work_data_cut = work_data_cut.reshape((work_data_cut.size,))
    work_data_cp[begin_orig_mask:end_orig_mask+1] = work_data_cut[:-block_diff] if block_remainder > 0 else work_data_cut

    return work_data_cp

def mark_time(idstr=None, last_time=None):
    new_time = time.time()

    if all_timing or (main_timing and idstr == 'measure_peaks.find_and_measure_peaks'):
        if last_time is not None:
            print idstr, "took ", (new_time - last_time), "to execute."

    return new_time

if __name__ == '__main__':
    main()

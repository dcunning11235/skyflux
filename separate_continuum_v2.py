import numpy as np
from astropy.table import Table
#from astropy.convolution import MexicanHat1DKernel
from astropy.convolution import Gaussian1DKernel
from astropy.convolution import convolve
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
noisy_sizes = np.array([15])
split_noisy_app = 2650 # = 6084.01 A

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

            ts = mark_time("getting match", ts)
            peaks = measure_peaks.find_and_measure_peaks(data, use_flux_con=False, ignore_defects=False)
            ts = mark_time("measure_peaks.find_and_measure_peaks", ts)
            peaks_mask = measure_peaks.mask_known_peaks(data, peaks)

            '''
            test_data = data.copy()
            test_data.mask = np.ma.nomask
            test_data.mask = [peaks_mask]*len(data.columns)
            peaks = measure_peaks.find_and_measure_peaks(test_data, use_flux_con=False, ignore_defects=False)
            '''

            #peaks_mask = np.zeros((7080,), dtype=bool)
            ts = mark_time("measure_peaks.mask_known_peaks", ts)
            data.mask = [orig_mask]*len(data.columns)

            start_continuum, start_wo_continuum = split_spectrum(data['wavelength'][:split_noisy_app], data['flux'][:split_noisy_app],
                                        peaks_mask[:split_noisy_app], orig_mask[:split_noisy_app],
                                        idstr=idstr, block_sizes=block_sizes)
            end_continuum, end_wo_continuum = split_spectrum(data['wavelength'][split_noisy_app:], data['flux'][split_noisy_app:],
                                        peaks_mask[split_noisy_app:], orig_mask[split_noisy_app:],
                                        idstr=idstr, block_sizes=block_sizes, mult=6)
            ts = mark_time("smoothing", ts)

            wo_continuum = np.ma.concatenate([start_wo_continuum, end_wo_continuum])
            continuum = np.ma.concatenate([start_continuum, end_continuum])

            #continuum, wo_continuum = tamp_down(continuum, wo_continuum, span=51)
            continuum, wo_continuum = tamp_down(continuum, wo_continuum)
            continuum, wo_continuum = tamp_down(continuum, wo_continuum, span=31)
            #continuum, wo_continuum = tamp_down(continuum, wo_continuum, span=31)
            continuum, wo_continuum = tamp_down(continuum, wo_continuum, span=21)
            #continuum, wo_continuum = tamp_down(continuum, wo_continuum, span=11)

            # Do not smooth
            #continuum, wo_continuum = smooth(continuum, wo_continuum)

            save_data(data['wavelength'], wo_continuum, continuum, data['ivar'], orig_mask, idstr)
            ts = mark_time("save_data", ts)

def tamp_down(continuum, wo_continuum, span=41):
    total = wo_continuum + continuum

    chunked_continuum, begin_orig_mask, end_orig_mask = trim_array_from_mask(continuum, continuum.mask, buffer=5)
    chunked_continuum.mask[begin_orig_mask:begin_orig_mask+5] = False
    chunked_continuum.mask[end_orig_mask-5:end_orig_mask] = False
    chunked_continuum, block_diff, block_remainder = chunk_array(chunked_continuum, begin_orig_mask,
                                                        end_orig_mask, span*3)
    chunked_continuum[:] = np.ma.mean(chunked_continuum, axis=1)[:, np.newaxis]
    chunked_continuum = chunked_continuum.reshape((chunked_continuum.size, ) )

    averages = np.zeros(continuum.size, dtype=float)
    averages[begin_orig_mask:end_orig_mask+1] = chunked_continuum[:-block_diff] if block_remainder > 0 else chunked_continuum

    def _moving_average(a, n=3):
        ret = np.cumsum(a, dtype=float)
        ret[n:] = ret[n:] - ret[:-n]
        ret = np.concatenate([a[:(n-1)/2], ret[n-1:]/n, a[-(n-1)/2:]])
        return ret

    move_avg_cont = _moving_average(continuum, n=span)
    too_loo_mask = move_avg_cont < averages

    move_avg_cont -= continuum
    move_avg_cont[move_avg_cont > 0] = 0
    move_avg_cont[too_loo_mask] = 0

    continuum += move_avg_cont
    wo_continuum = total - continuum

    return continuum, wo_continuum

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

def split_spectrum(work_wlen, work_data, peaks_mask, orig_mask, block_sizes, idstr=None, mult=1):
    work_wlen_cp = work_wlen.copy()
    work_data_cp = work_data.copy()

    work_data_cp = np.ma.mean( [kill_peaks(work_wlen_cp, work_data_cp, peaks_mask, orig_mask, block,
                                    cutoff=cutoff, is_noisy=True) for block, cutoff in
                                    zip(noisy_sizes*mult, noisy_cutoffs/mult)], axis=0 )
    '''
    plt.plot(work_wlen, work_data_cp)
    plt.tight_layout()
    plt.show()
    plt.close()
    '''

    work_data_cp = np.ma.mean( [kill_peaks(work_wlen_cp, work_data_cp, peaks_mask, orig_mask,
                                    block, cutoff=cutoff) for block, cutoff in
                                    zip(block_sizes, base_stds)], axis=0 )
    '''
    plt.plot(work_wlen, work_data_cp)
    plt.tight_layout()
    plt.show()
    plt.close()
    '''
    
    cont_flux = work_data_cp
    return cont_flux, work_data-cont_flux

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
        data = np.concatenate([data, extent])
    data = data.reshape(new_shape)

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

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

import stack
import measure_peaks

# Magic numbers.... oooohhhhhhhh!
block_sizes = np.array([8,12,20])
base_stds = np.array([0.7, 0.9, 1])
noisy_cutoffs = np.array([1.85])
noisy_sizes = np.array([60])
split_noisy_app = 2640

all_timing = False
main_timing = False
ts = time.time()

# TBD: Old notes removed; need to write up new notes on this

# Bugs, Issues:
#  1.)  Fills all con_flux slots, even those that are masked; these should not be
#       output to the csv file, even if they are used internally.  Low priority,
#       the ivar value is what matters as far as masking goes.
# 2.)  This used to be super-fast, now is super-slow.  Is it the cwt?  Is it the
#       peak filtering? (probably)  Something else?

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
            peaks = measure_peaks.find_and_measure_peaks(data, use_flux_con=False)
            ts = mark_time("measure_peaks.find_and_measure_peaks", ts)
            peaks_mask = measure_peaks.mask_known_peaks(data, peaks)
            #peaks_mask = np.zeros((7080,), dtype=bool)
            ts = mark_time("measure_peaks.mask_known_peaks", ts)
            data.mask = [orig_mask]*len(data.columns)

            start_continuum, start_wo_continuum = smoothing(data['wavelength'][:split_noisy_app], data['flux'][:split_noisy_app],
                                        peaks_mask[:split_noisy_app], orig_mask[:split_noisy_app],
                                        idstr=idstr, block_sizes=block_sizes)
            end_continuum, end_wo_continuum = smoothing(data['wavelength'][split_noisy_app:], data['flux'][split_noisy_app:],
                                        peaks_mask[split_noisy_app:], orig_mask[split_noisy_app:],
                                        idstr=idstr, block_sizes=block_sizes, mult=2)
            ts = mark_time("smoothing", ts)

            wo_continuum = np.ma.concatenate([start_wo_continuum, end_wo_continuum])
            continuum = np.ma.concatenate([start_continuum, end_continuum])
            continuum, wo_continuum = tamp_down(continuum, wo_continuum)

            continuum, wo_continuum = smooth(continuum, wo_continuum)

            save_data(data['wavelength'], wo_continuum, continuum, data['ivar'], orig_mask, idstr)
            ts = mark_time("save_data", ts)

def tamp_down(continuum, wo_continuum):
    total = wo_continuum + continuum

    def _moving_average(a, n=3):
        ret = np.cumsum(a, dtype=float)
        ret[n:] = ret[n:] - ret[:-n]
        ret = np.concatenate([a[:(n-1)/2], ret[n-1:]/n, a[-(n-1)/2:]])
        return ret

    move_avg_cont = _moving_average(continuum, n=31)
    move_avg_cont -= continuum
    move_avg_cont[move_avg_cont > 0] = 0

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

def smoothing(work_wlen, work_data, peaks_mask, orig_mask, block_sizes, idstr=None, mult=1):
    work_wlen_cp = work_wlen.copy()
    work_data_cp = work_data.copy()

    work_data_cp = np.ma.mean( [kill_peaks(work_wlen_cp, work_data_cp, peaks_mask, orig_mask, block,
                                    cutoff=cutoff, is_noisy=True) for block, cutoff in
                                    zip(noisy_sizes*mult, noisy_cutoffs)], axis=0 )
    work_data_cp = np.ma.mean( [kill_peaks(work_wlen_cp, work_data_cp, peaks_mask, orig_mask,
                                    block, cutoff=cutoff) for block, cutoff in
                                    zip(block_sizes, base_stds)], axis=0 )

    #work_data_cp = np.ma.mean( [kill_peaks(work_data, block) for block in block_sizes], axis=0 )

    cont_flux = work_data_cp
    return cont_flux, work_data-cont_flux

def kill_peaks(work_wlen_cp, work_data_cp, peaks_mask, orig_mask, block_size, block_offset=0,
                cutoff=None, is_noisy=False):
    data_len = len(work_data_cp)
    orig_mask_extents = np.where(~orig_mask)
    begin_orig_mask = np.min(orig_mask_extents)
    end_orig_mask = np.max(orig_mask_extents)
    overall_average = np.ma.mean(work_data_cp)

    work_data_cp[:begin_orig_mask] = overall_average
    work_data_cp[end_orig_mask:] = overall_average

    combined_mask = peaks_mask | orig_mask

    #This unmaskes everything...
    work_data_cp[combined_mask] = np.interp(work_wlen_cp[combined_mask], work_wlen_cp[~combined_mask], work_data_cp[~combined_mask])

    block_diff = data_len % block_size
    working_slice = slice(None, None)
    new_shape = ( (data_len-block_diff)/block_size, block_size)
    working_slice = slice(0, data_len-block_diff)
    leftovers = work_data_cp[-block_diff:]
    work_data_cp = work_data_cp[working_slice].reshape(new_shape)

    combined_mask = combined_mask[working_slice].reshape(new_shape)
    masked_count = np.ma.sum(combined_mask, axis=1)
    stdevs = np.ma.std(work_data_cp, axis=1)
    overs = np.ma.min(work_data_cp, axis=1)
    ins = np.ma.mean(work_data_cp, axis=1)

    stdevs[masked_count > (block_size/3)] = 0
    x = np.arange(len(stdevs))
    mask = (stdevs == 0) & (overs == 0)

    stdevs[mask] = np.interp(x[mask], x[~mask], stdevs[~mask])
    overs[mask] = np.interp(x[mask], x[~mask], overs[~mask])
    ins[mask] = np.interp(x[mask], x[~mask], ins[~mask])

    new_work_data = work_data_cp.copy()

    if not is_noisy:
        stdev_rows = np.where(stdevs <= cutoff)
        new_work_data[stdev_rows,:] = ins[stdev_rows,np.newaxis]
        stdev_rows = np.where(stdevs > cutoff)
        new_work_data[stdev_rows,:] = overs[stdev_rows,np.newaxis]
    else:
        stdev_rows = np.where((stdevs > cutoff) | (masked_count > (block_size/3)))
        new_work_data[stdev_rows,:] = overs[stdev_rows,np.newaxis]

    #print "new_work_data size:", new_work_data.size
    new_work_data = new_work_data.reshape((new_work_data.size,))
    new_work_data.mask = work_data_cp.mask
    if block_diff:
        #print "new_work_data size + block_diff:", new_work_data.size + block_diff
        temp = np.ma.empty((new_work_data.size + block_diff, ), dtype=float)
        #print "temp shape:", temp.shape
        temp[0:new_work_data.size] = new_work_data
        temp[new_work_data.size:] = leftovers
        new_work_data = temp

    #print "Final output shape:", new_work_data.shape
    return new_work_data

def mark_time(idstr=None, last_time=None):
    new_time = time.time()

    if all_timing or (main_timing and idstr == 'measure_peaks.find_and_measure_peaks'):
        if last_time is not None:
            print idstr, "took ", (new_time - last_time), "to execute."

    return new_time

if __name__ == '__main__':
    main()

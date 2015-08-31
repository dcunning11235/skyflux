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

block_sizes = [8,12,20]
base_stds = [0.4, 0.6, 0.75]

noisy_cutoffs = [1.2, 1.4]
#noisy_cutoffs = [1.55, 1.8] #, 1.75]
noisy_sizes = [30, 40] #, 60]

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
            ts = mark_time("measure_peaks.mask_known_peaks", ts)
            data.mask = [orig_mask]*len(data.columns)

            continuum, wo_continuum = smoothing(data['wavelength'], data['flux'], peaks_mask,
                                        orig_mask, idstr=idstr, keep_chunky=False, block_sizes=block_sizes)
            ts = mark_time("smoothing", ts)

            save_data(data['wavelength'], wo_continuum, continuum, data['ivar'], orig_mask, idstr)
            ts = mark_time("save_data", ts)

            del(peaks_mask)
            del(peaks)

def save_data(wlen, flux, con_flux, ivar, mask, idstr):
    wlen.mask = np.ma.nomask
    flux.mask = mask
    con_flux = np.ma.array(con_flux, mask=mask)
    #con_flux.mask = mask
    ivar.mask = mask
    continuum_table = Table([wlen.data, flux.filled(0), con_flux.filled(0), ivar.filled(0)], names=["wavelength", "flux", "con_flux", "ivar"])
    continuum_table.write("{}-continuum.csv".format(idstr), format="ascii.csv")

def smoothing(work_wlen, work_data, peaks_mask, orig_mask, block_sizes, idstr=None, keep_chunky=False):
    work_wlen_cp = work_wlen.copy()
    work_data_cp = work_data.copy()

    work_data_cp = np.ma.mean( [kill_peaks(work_wlen_cp, work_data_cp, peaks_mask, orig_mask, block,
                                    cutoff=cutoff, is_noisy=True) for block, cutoff in
                                    zip(noisy_sizes, noisy_cutoffs)], axis=0 )
    work_data_cp = np.ma.mean( [kill_peaks(work_wlen_cp, work_data_cp, peaks_mask, orig_mask,
                                    block, cutoff=cutoff) for block, cutoff in
                                    zip(block_sizes, base_stds)], axis=0 )

    #work_data_cp = np.ma.mean( [kill_peaks(work_data, block) for block in block_sizes], axis=0 )

    if not keep_chunky:
        g = Gaussian1DKernel(stddev=8)
        #g = MexicanHat1DKernel(width=10)
        z = convolve(work_data_cp, g, boundary='extend')

    cont_flux = work_data_cp if keep_chunky else z

    return cont_flux, work_data-cont_flux

def kill_peaks(work_wlen_cp, work_data_cp, peaks_mask, orig_mask, block_size, block_offset=0,
                cutoff=None, is_noisy=False):
    orig_mask_extents = np.where(~orig_mask)
    begin_orig_mask = np.min(orig_mask_extents)
    end_orig_mask = np.max(orig_mask_extents)
    overall_average = np.ma.mean(work_data_cp)

    work_data_cp[:begin_orig_mask] = overall_average
    work_data_cp[end_orig_mask:] = overall_average

    combined_mask = peaks_mask | orig_mask
    #combined_mask[:begin_orig_mask] = False
    #combined_mask[end_orig_mask:] = False

    #This unmaskes everything...
    work_data_cp[combined_mask] = np.interp(work_wlen_cp[combined_mask], work_wlen_cp[~combined_mask], work_data_cp[~combined_mask])

    combined_mask = combined_mask.reshape( ( (len(work_data_cp)-block_offset)/block_size, block_size) )
    masked_count = np.ma.sum(combined_mask, axis=1)

    work_data_cp = work_data_cp.reshape(( (len(work_data_cp)-block_offset)/block_size, block_size))
    #masked_count = np.ma.count_masked(work_data_cp, axis=1)
    stdevs = np.ma.std(work_data_cp, axis=1)
    overs = np.ma.min(work_data_cp, axis=1)
    ins = np.ma.mean(work_data_cp, axis=1)

    stdevs[masked_count > (block_size/3)] = 0
    x = np.arange(len(stdevs))
    mask = (stdevs == 0) & (overs == 0)

    #begin_block_mask = begin_orig_mask // block_size
    #end_block_mask = end_orig_mask // block_size
    #mask[:begin_block_mask] = False
    #mask[end_block_mask:] = False

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

        #if np.any(begin_block_mask == stdev_rows):
        #    new_work_data[begin_block_mask] = ins[begin_block_mask]
        #    new_work_data[begin_block_mask+1] = ins[begin_block_mask+1]

    new_work_data = new_work_data.reshape((new_work_data.size,))
    new_work_data.mask = work_data_cp.mask

    return new_work_data

def mark_time(idstr=None, last_time=None):
    new_time = time.time()

    if all_timing or (main_timing and idstr == 'measure_peaks.find_and_measure_peaks'):
        if last_time is not None:
            print idstr, "took ", (new_time - last_time), "to execute."

    return new_time

if __name__ == '__main__':
    main()

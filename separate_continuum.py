import numpy as np
from astropy.table import Table
import fnmatch
import os
import os.path
import sys

import stack

block_sizes = [8,12,20,120]
base_stds = [0.4, 0.6, 0.9, 1.1]

noisy_cutoffs = [1.75, 2.5, 3]
noisy_sizes = [30, 40, 120]

# There are several issues with this implementation/algorithm
#
# 1.)  There is a trade off between finding the base/continua of very nosiy, "peaky" regions
#           and over-flatenning absorption features.  This mainly has to do with the size
#           of the 'noisy_sizes' setting; wider flattens more, eliminatting peaks but
#           erasing absorption featurs.  Possible solution:  Add 'noisy_cutoff' list and
#           further tune STD cutoffs.   <----- FIX THIS
#           (UPDATE: Tuned further, values added; this can go on forever, and will never be
#           perfect)
# 2.)  The above is a problem; I believe the peak at 7468.2 (N) is being affected by this in
#           the 3586 exposures; I think this may be because of the above, and/or...
# 3.)  In cases where the STD window is masked there are two issues:
#               a.)  Not taking enough values into consideration; this is mostly a 'pure'
#                   statistical issue.   <----- FIX THIS
#                   (UPDATE: semi-address)
#               b.)  The lowest point calculation can be skewed (severly) if e.g. the only
#                   non-masked value are part of a peak.  Need to implement some logic to
#                   extend the region to include non-masked values, or just reject the region
#                   and interp. between neigboring regions... <-------- FIX THIS
#                   (UPDATE:  semi-addressed)
# 4.)  There is a slight bias between regions:  in noisy regions the lowest point is sought,
#           in non-noisy regions over the (small) 'base_std' the lowest point is sought,
#           and in very quiet regions the average is sought.  So over the e.g. 4500-5500 (very
#           roughly, from memory) region the baseline is 'shifted up' by something like half a
#           flux unit.  This is small, but potentially important in comparing small features.
#           Could relatively easily emperically test for the actual difference, if it comes to
#           that.

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
            data.mask = [(data['flux'] == 0)]*len(data.columns)
            idstr = file[:file.rfind('.')]

            print file
            continuum, wo_continuum = smoothing(data['wavelength'],
                data['flux'], data.mask, idstr=idstr, keep_chunky=True, block_sizes=block_sizes)
            save_data(data['wavelength'], wo_continuum, continuum, data['ivar'], idstr)

def save_data(wlen, flux, con_flux, ivar, idstr):
    wlen.mask = np.ma.nomask
    continuum_table = Table([wlen.data, flux.filled(0), con_flux.filled(0), ivar.filled(0)], names=["wavelength", "flux", "con_flux", "ivar"])
    continuum_table.write("{}-continuum.csv".format(idstr), format="ascii.csv")

def smoothing(work_wlen, work_data, orig_mask, block_sizes, idstr=None, keep_chunky=False):
    work_data_cp = np.ma.mean( [kill_peaks(work_wlen, work_data, block, cutoff=cutoff, is_noisy=True) for block, cutoff in zip(noisy_sizes, noisy_cutoffs)], axis=0 )
    work_data_cp = np.ma.mean( [kill_peaks(work_wlen, work_data_cp, block, cutoff=cutoff) for block, cutoff in zip(block_sizes, base_stds)], axis=0 )

    #work_data_cp = np.ma.mean( [kill_peaks(work_data, block) for block in block_sizes], axis=0 )

    if not keep_chunky:
        g = Gaussian1DKernel(stddev=10)
        z = convolve(work_data_cp, g, boundary='extend')

    cont_flux = work_data_cp if keep_chunky else z

    return cont_flux, work_data-cont_flux

def kill_peaks(work_wlen, work_data, block_size, block_offset=0, cutoff=None, is_noisy=False):
    work_data_cp = work_data[block_offset:]
    '''
    work_wlen_cp = work_wlen[block_offset:]

    int_filled = np.arange(len(work_data_cp))
    mask_arr = work_data_cp.mask.copy()
    work_data_cp.mask = np.ma.nomask
    work_wlen_cp.mask = np.ma.nomask
    work_wlen_cp[mask_arr] = np.interp(int_filled[mask_arr], int_filled[~mask_arr], work_wlen_cp[~mask_arr])
    work_data_cp[mask_arr] = np.interp(work_wlen_cp[mask_arr], work_wlen_cp[~mask_arr], work_data_cp[~mask_arr])

    print work_data_cp[5231:5281]
    print np.ma.count_masked(work_data_cp)
    '''

    work_data_cp = work_data_cp.reshape(( (len(work_data)-block_offset)/block_size, block_size))
    masked_count = np.ma.count_masked(work_data_cp, axis=1)
    stdevs = np.ma.std(work_data_cp, axis=1)
    overs = np.ma.min(work_data_cp, axis=1)
    ins = np.ma.mean(work_data_cp, axis=1)

    stdevs[masked_count > (block_size/3)] = 0

    x = np.arange(len(stdevs))
    mask = (stdevs == 0)
    stdevs[mask] = np.interp(x[mask], x[~mask], stdevs[~mask])
    overs[mask] = np.interp(x[mask], x[~mask], overs[~mask])
    ins[mask] = np.interp(x[mask], x[~mask], ins[~mask])

    block_start = np.rint(stack.get_stacked_fiducial_wlen_pixel_offset(8600)) // block_size

    n = block_size
    new_work_data = work_data_cp.copy()

    if not is_noisy:
        if cutoff is None:
            cutoff = base_stds[0]
        stdev_rows = np.where(stdevs <= cutoff)
        new_work_data[stdev_rows,:] = ins[stdev_rows,np.newaxis]
        stdev_rows = np.where(stdevs > cutoff)
        new_work_data[stdev_rows,:] = overs[stdev_rows,np.newaxis]
    else:
        if cutoff is None:
            cutoff = noisy_cutoffs[0]
        stdev_rows = np.where(stdevs > cutoff)
        new_work_data[stdev_rows,:] = overs[stdev_rows,np.newaxis]

    new_work_data = new_work_data.reshape((new_work_data.size,))
    if new_work_data.size < len(work_data):
        temp = np.ma.empty_like(work_data)
        temp[block_offset:] = new_work_data
        temp[:block_offset] = new_work_data[0]
    new_work_data.mask = work_data_cp.mask

    return new_work_data

if __name__ == '__main__':
    main()

import numpy as np

from astropy.table import Table
from astropy.table import Column
from astropy.table import vstack
from astropy.utils.compat import argparse

import scipy.integrate as intg
from scipy.signal import find_peaks_cwt

import fnmatch
import os
import os.path
import sys

import stack

import numpy.lib.recfunctions as rfn

# New method for peak detection using scipy.signal.find_peaks_cwt.  Have, for now at least
# completely stopped using these line_marker dicts at the top of the file.  What I should do
# is get a database of emission lines for the key atomic peaks:  N, Na, Li, Hg, O, etc. and
# have a last step before output be a matching step where peaks are matched to the their
# closest (within some range) matching emission line.  Where can I get OH, CO2, H2O lines?

line_markers = {
    "Ar": [5087.1, 8849.9], #Mayyyybe?  Or Krypton?  Or something...
    "Li": [3746.58, 3796.1, 3915.3, 4273.1],
    "Li-II": [5199.3], #avg of many close lines
    "Hg": [3650.2, 3654.8, 3663.3, 4046.6, 4358.3, 5460.8, 5675.9, 5769.6, 5790.7, 7602.2],
    "Na": [3881.8, 4982.8, 5889.9, 5895.9],
    "O": [6156.0, 6300.3, 7002.2, 7254.2, 7254.5],
    "N": [5199.8, 5200.3, 7306.6, 7468.3, 9386.8],
    "N-II": [5199.5],
    "Hg-II": [5871.3, 5888.9, 6146.4, 6521.1],
    "O-II": [7599.2],
    "O-III": [3715.1],
    "Kr-II": [5690.4] #Or... Na-VII @ 5690
}

questionable_to_bad_line_markers = {
    "C-III": [8367.9],
    "N": [7442.3],
    "Hg": [6907.5],
    "Hg-II": [5204.8, 7346.6]
}

bad_line_markers = {
    "O": [6156.8, 6158.2, 6456],
    "N": [5752.5, 7423.6],
    "Hg": [3550.2, 5803.7, 6716.3],
    "Li": [3671.7, 3720.9, 3985.5, 4132.6, 4602.8, 4917.7, 6103.5],
    "Na-II": [3533.1, 3631.2, 3711.1, 3858.3],
    "Hg-II": [3532.6, 3605.8, 3989.3, 5128.4, 5425.3, 5595.3, 5677.1, 6149.5]
}

solar_absorb_line_markers = {
    "Ca-II": [3968, 3934],
    "H": [4861, 6563],
    "Fe": [5270]
    #"Na": [5889.9, 5895.9] #as above
}
wlen_spans = {
    "blue": [(3800,5000)],
    "mid": [(5000,7200)],
    "red": [(7200,10300)],
    "blue_half": [(3800,6400)],
    "red_half": [(6400,10350)]
}

total_dtype=[('total_type', object), ('source', object), ('wavelength_target', float),
            ('wavelength_lower_bound', float), ('index_lower_bound', int),
            ('wavelength_upper_bound', float), ('index_upper_bound', int),
            ('wavelength_peak', float), ('peak_delta', float),
            ('peak_delta_over_width', float), ('total_flux', float), ('total_con_flux', float)]

max_peak_width = 20
peak_widths = [3,7,max_peak_width]

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
            data['wavelength'].mask = False
            idstr = file[:file.rfind('.')]

            peak_flux_list = []
            peak_flux = find_and_measure_peaks(data, peak_flux_list)

            #Let's just forget this for now; spans are maybe something to come back to
            '''
            for key, vals in wlen_spans.items():
                target_flux_totals = get_total_flux(key, data['wavelength'], data['flux'], data['con_flux'], target_wlens=None, wlen_spans=vals)
                peak_flux_list.append(target_flux_totals)
            '''

            save_data(peak_flux, idstr)

def find_and_measure_peaks(data, peak_flux_list=[], use_flux_con=True):
    arr = []
    if len(peak_flux_list) > 0:
        arr = rfn.stack_arrays(peak_flux_list)

    found_peaks, found_inds = real_find_peaks(data)
    removed = False

    for candidate_peak, candidate_ind in zip(found_peaks, found_inds):
        removed = False
        if candidate_peak is np.ma.masked:
            continue
        for peak in arr:
            if (candidate_peak > peak['wavelength_lower_bound'] and
                    candidate_peak < peak['wavelength_upper_bound']  and
                    np.abs(candidate_ind - peak['index_lower_bound']) >= max_peak_width and
                    np.abs(candidate_ind - peak['index_upper_bound']) >= max_peak_width):
                #found_peaks.remove(peak)
                removed=True
                break
        if ~removed:
            target_flux_totals = get_total_flux("UNKNOWN", data['wavelength'], data['flux'],
                                None if not use_flux_con else data['con_flux'], candidate_peak)
            peak_flux_list.append(target_flux_totals)

    #Now, need to prune the list
    arr = rfn.stack_arrays(peak_flux_list)
    peak_flux = Table(data=arr)

    peak_flux.remove_rows(np.abs(peak_flux['peak_delta']) > max_peak_width)
    peak_flux = filter_for_overlaps(peak_flux, ['index_lower_bound', 'index_upper_bound'])
    peak_flux = filter_for_overlaps(peak_flux, ['index_lower_bound'])
    peak_flux = filter_for_overlaps(peak_flux, ['index_upper_bound'])

    return peak_flux

def filter_for_overlaps(peak_flux, cols):
    peak_flux_filtered_list = []

    group_peak_flux = peak_flux.group_by(cols)
    for group in group_peak_flux.groups:
        if len(group) > 1:
            min_peak_delta = np.min(np.abs(group['peak_delta']))
            peak_flux_filtered_list.append(group[group['peak_delta'] == min_peak_delta])
        else:
            peak_flux_filtered_list.append(group[0])
    peak_flux_filtered_arr = rfn.stack_arrays(peak_flux_filtered_list)
    peak_flux = Table(data=peak_flux_filtered_arr)

    return peak_flux

def save_data(peak_flux, idstr):
    peak_flux.write('{}-peaks.csv'.format(idstr), format='ascii.csv')

def mask_range(data, start_ind, end_ind):
    cols = [name for name in data.colnames if name != 'wavelength']
    for col in cols:
        data[col].mask[start_ind:end_ind+1] = True

def mask_known_peaks(data, peaks):
    peaks_mask = np.zeros( (len(data), ), dtype=bool)

    for peak in peaks:
        if peak['index_upper_bound'] != 0:
            mask_range(data, peak['index_lower_bound'], peak['index_upper_bound'])
            peaks_mask[peak['index_lower_bound']:peak['index_upper_bound']+1] = True

    return peaks_mask

def real_find_peaks(data,cols=['flux']):
    val = data[cols[0]]
    if len(cols) > 1:
        for col_name in cols[1:]:
            val += data[col_name]
    peak_inds = find_peaks_cwt(val, np.array(peak_widths), noise_perc=5)
    peaks = []
    for ind in peak_inds:
        peaks.append(data['wavelength'][ind])
    return peaks, peak_inds

def get_total_flux(label, wlen, flux, con_flux, target_wlens=None, wlen_spans=None):
    #old = np.seterr(all='raise')

    ret_len = 0
    if target_wlens is not None:
        if not hasattr(target_wlens, "__iter__"):
            target_wlens = [target_wlens]
        ret_len += len(target_wlens)
    if wlen_spans is not None:
        ret_len += len(wlen_spans)
    ret = np.ndarray(ret_len, dtype=total_dtype)
    ind = 0

    if target_wlens is not None:
        for target in target_wlens:
            offset = stack.get_stacked_fiducial_wlen_pixel_offset(target)
            offset_val = flux[offset]
            under_offset = over_offset = offset
            peak = False

            while under_offset > 0 and (flux[under_offset-1] <= flux[under_offset] or not peak):
                if flux[under_offset-1] <= flux[under_offset]:
                    peak = True
                under_offset -= 1

            peak = False
            while under_offset > 0 and over_offset < (len(wlen)-1) and (flux[over_offset+1] <= flux[over_offset] or not peak):
                if flux[over_offset+1] <= flux[over_offset]:
                    peak = True
                over_offset += 1

            if under_offset < 1 or over_offset > len(wlen)-2:
                #print "ran off end of spectrum with (under_offset, over_offset) = ", (under_offset, over_offset)
                ret[ind] = ('line', label, target, 0, 0, 0, 0, 0, 0, 0, 0, 0)
                ind += 1
                continue

            #Now, try to account for cases where peaks are jagged and we're stuck on a small dip
            if (np.ma.min( flux[max(0, under_offset-(max_peak_width//2)):under_offset]) < (flux[under_offset] / 2)) and \
                    (np.ma.max( flux[max(0, under_offset-(max_peak_width//2)):under_offset+1]) < np.ma.max(flux[under_offset:over_offset+1]) / 5):
                #Start block
                under_under_offset = np.ma.argmin( flux[max(0, under_offset-(max_peak_width//2)):under_offset+1])
                new_under_offset = under_offset - ((max_peak_width//2)-under_under_offset)
                under_offset = new_under_offset
            if (np.ma.min( flux[over_offset+1:min(over_offset+(max_peak_width//2), len(wlen))]) < (flux[over_offset] / 2)) and \
                    (np.ma.max( flux[over_offset+1:min(over_offset+(max_peak_width//2), len(wlen))]) < np.ma.max(flux[under_offset:over_offset+1]) / 5):
                #Start block
                over_over_offset = np.ma.argmin( flux[over_offset+1:min(over_offset+(max_peak_width//2), len(wlen))])
                new_over_offset = over_offset + over_over_offset
                over_offset = new_over_offset

            nonmasked_start = int(np.ma.notmasked_edges(flux[:under_offset+1])[1])
            nonmasked_end = int(np.ma.notmasked_edges(flux[over_offset:])[0])

            flux_filled = flux[nonmasked_start:over_offset+nonmasked_end+1]
            if con_flux is not None:
                con_flux_filled = con_flux[nonmasked_start:over_offset+nonmasked_end+1]
            wlen_filled = wlen[nonmasked_start:over_offset+nonmasked_end+1]
            int_filled = np.arange(len(wlen_filled))
            mask_arr = wlen_filled.mask.copy()
            wlen_filled.mask = np.ma.nomask
            flux_filled.mask = np.ma.nomask

            try:
                wlen_filled[mask_arr] = np.interp(int_filled[mask_arr], int_filled[~mask_arr], wlen_filled[~mask_arr])
                flux_filled[mask_arr] = np.interp(wlen_filled[mask_arr], wlen_filled[~mask_arr], flux_filled[~mask_arr])
                if con_flux is not None:
                    con_flux_filled[mask_arr] = np.interp(wlen_filled[mask_arr], wlen_filled[~mask_arr], con_flux_filled[~mask_arr])

                start_ind = under_offset - nonmasked_start
                end_ind = over_offset+1 - nonmasked_end
                if end_ind == 0:
                    end_ind=None

                total = intg.simps(flux_filled[start_ind:end_ind], wlen_filled[start_ind:end_ind])
                con_total = 0
                if con_flux is not None:
                    con_total = intg.simps(con_flux_filled[start_ind:end_ind], wlen_filled[start_ind:end_ind])
                range_start = wlen[under_offset]
                range_end = wlen[over_offset]
                range_peak = wlen_filled[flux_filled.argsort()[-1]]
                peak_delta = (range_peak - target)
                ret[ind] = ("line", label, target, range_start, under_offset, range_end,
                            over_offset, range_peak, peak_delta,
                            peak_delta/(range_end - range_start), total, con_total)
                ind += 1
            except:
                print(int_filled)
                print(flux_filled)
                print(wlen_filled)
                print(flux_filled[~mask_arr])
                print(flux_filled[mask_arr])
                print(wlen_filled[~mask_arr])
                print(wlen_filled[mask_arr])
                raise

    if wlen_spans is not None:
        for i, span in enumerate(wlen_spans):
            start_offset = stack.get_stacked_fiducial_wlen_pixel_offset(span[0])
            end_offset = stack.get_stacked_fiducial_wlen_pixel_offset(span[1])

            total = intg.simps(flux[start_offset:end_offset+1], wlen[start_offset:end_offset+1])
            con_total = intg.simps(con_flux[start_offset:end_offset+1], wlen[start_offset:end_offset+1])
            ret[ind] = ("span", label, 0, wlen[start_offset], start_offset, wlen[end_offset], end_offset, 0, 0, 0, total, con_total)
            ind += 1

    return ret

if __name__ == '__main__':
    main()

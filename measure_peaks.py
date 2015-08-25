import numpy as np

from astropy.table import Table
from astropy.table import Column
from astropy.table import vstack
from astropy.utils.compat import argparse

import scipy.integrate as intg

import fnmatch
import os
import os.path
import sys

import stack

import numpy.lib.recfunctions as rfn

line_markers = {
    "Li": [3671.7, 3720.9, 3746.58, 3796.1, 3915.3, 3985.5, 4132.6, 4273.1, 4602.8, 4917.7, 6103.5],
    "Li-II": [5199.3], #avg of many close lines
    "Hg": [3550.2, 3654.8, 4046.6, 4358.3, 5460.8, 5675.9, 5769.6, 5790.7, 5803.7, 6716.3, 6907.5, 7602.2],
    "Na": [3881.8, 4982.8, 5889.9, 5895.9],
    "O": [6156.0, 6156.8, 6158.2, 6456, 7002.2, 7254.2, 7254.5],
    "N": [5199.8, 5200.3, 5752.5, 7306.6, 7423.6, 7442.3, 7468.3],
    "N-II": [5199.5],
    "Na-II": [3533.1, 3631.2, 3711.1, 3858.3],
    "Hg-II": [3532.6, 3605.8, 3989.3, 5128.4, 5204.8, 5425.3, 5595.3, 5677.1, 5871.3, 5888.9, 6146.4, 6149.5, 6521.1, 7346.6],
    "O-II": [7599.2]
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
            data['wavelength'].mask = False
            idstr = file[:file.rfind('.')]

            total_dtype=[('total_type', object), ('source', object), ('wavelength_target', float), ('wavelength_lower_bound', float),
                        ('wavelength_upper_bound', float), ('total_flux', float), ('total_con_flux', float)]
            peak_flux_list = []
            for key, vals in line_markers.items():
                target_flux_totals = get_total_flux(key, data['wavelength'], data['flux'], data['con_flux'], vals)
                peak_flux_list.append(target_flux_totals)
            for key, vals in wlen_spans.items():
                target_flux_totals = get_total_flux(key, data['wavelength'], data['flux'], data['con_flux'], target_wlens=None, wlen_spans=vals)
                peak_flux_list.append(target_flux_totals)
            save_data(peak_flux_list, idstr)

def save_data(peak_flux_list, idstr):
    arr = rfn.stack_arrays(peak_flux_list)
    peak_flux = Table(data=arr)
    peak_flux.write('{}-peaks.csv'.format(idstr), format='ascii.csv')

def get_total_flux(label, wlen, flux, con_flux, target_wlens=None, wlen_spans=None):
    old = np.seterr(all='raise')

    total_dtype=[('total_type', object), ('label', object), ('wavelength_target', float), ('wavelength_lower_bound', float),
                ('wavelength_upper_bound', float), ('total_flux', float), ('total_con_flux', float)]

    ret_len = 0
    if target_wlens is not None:
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
                ret[ind] = ('line', label, 0, 0, 0, 0, 0)
                ind += 1
                continue

            #Now, try to account for cases where peaks are jagged and we're stuck on a small dip
            if (np.ma.min( flux[max(0, under_offset-4):under_offset] - con_flux[max(0, under_offset-4):under_offset]) <
                    (flux[under_offset] - con_flux[under_offset]) / 2) and \
                (np.ma.max( flux[max(0, under_offset-4):under_offset+1] - con_flux[max(0, under_offset-4):under_offset+1]) <
                    np.ma.max(flux[under_offset:over_offset+1] - con_flux[under_offset:over_offset+1]) / 5):
                #Start block
                under_under_offset = np.ma.argmin( flux[max(0, under_offset-4):under_offset+1] - con_flux[max(0, under_offset-4):under_offset+1])
                new_under_offset = under_offset - (4-under_under_offset)
                under_offset = new_under_offset
            if (np.ma.min( flux[over_offset+1:min(over_offset+5, len(wlen))] - con_flux[over_offset+1:min(over_offset+5, len(wlen))]) <
                    (flux[over_offset] - con_flux[over_offset]) / 2) and \
                (np.ma.max( flux[over_offset+1:min(over_offset+5, len(wlen))] - con_flux[over_offset+1:min(over_offset+5, len(wlen))]) <
                    np.ma.max(flux[under_offset:over_offset+1] - con_flux[under_offset:over_offset+1]) / 5):
                #Start block
                over_over_offset = np.ma.argmin( flux[over_offset+1:min(over_offset+5, len(wlen))] - con_flux[over_offset+1:min(over_offset+5, len(wlen))])
                new_over_offset = over_offset + over_over_offset
                over_offset = new_over_offset

            nonmasked_start = int(np.ma.notmasked_edges(flux[:under_offset+1])[1])
            nonmasked_end = int(np.ma.notmasked_edges(flux[over_offset:])[0])

            flux_filled = flux[nonmasked_start:over_offset+nonmasked_end+1]
            con_flux_filled = con_flux[nonmasked_start:over_offset+nonmasked_end+1]
            wlen_filled = wlen[nonmasked_start:over_offset+nonmasked_end+1]
            int_filled = np.arange(len(wlen_filled))
            mask_arr = wlen_filled.mask.copy()
            wlen_filled.mask = np.ma.nomask
            flux_filled.mask = np.ma.nomask

            try:
                wlen_filled[mask_arr] = np.interp(int_filled[mask_arr], int_filled[~mask_arr], wlen_filled[~mask_arr])
                flux_filled[mask_arr] = np.interp(wlen_filled[mask_arr], wlen_filled[~mask_arr], flux_filled[~mask_arr])
                con_flux_filled[mask_arr] = np.interp(wlen_filled[mask_arr], wlen_filled[~mask_arr], con_flux_filled[~mask_arr])

                start_ind = under_offset - nonmasked_start
                end_ind = over_offset+1 - nonmasked_end
                if end_ind == 0:
                    end_ind=None
                total = intg.simps(flux_filled[start_ind:end_ind], wlen_filled[start_ind:end_ind])
                con_total = intg.simps(con_flux_filled[start_ind:end_ind], wlen_filled[start_ind:end_ind])
                ret[ind] = ("line", label, target, wlen[under_offset], wlen[over_offset], total, con_total)
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
            ret[ind] = ("span", label, 0, wlen[start_offset], wlen[end_offset], total, con_total)
            ind += 1

    return ret

if __name__ == '__main__':
    main()

import numpy as np
from astropy.table import Table
from astropy.table import vstack
import astropy.coordinates as ascoord
import sys
import datetime as dt
import bossdata.path as bdpath
import bossdata.remote as bdremote
import bossdata.spec as bdspec

'''
This script takes in a list of plate-mjd combos (as output by e.g. bossquery for
    bossquery --what "PLATE,MJD" --what "PLATE<=3650 and FIBER=1" --save some_plates.dat

It walks the list and outputs metadata about the exposures that make up those plates; it does
this by looking at metadata in (full) spec files for fibers 1 and 501.  Since there there
should (must) be a red and a blue exposure for each full exposure, corresponing color exposures
are output on one line.
'''

tai_base_date_time = dt.datetime(1858, 11, 17)
finder = bdpath.Finder()
manager = bdremote.Manager()

def file_deets(plate, mjd, spectrograph, gather=False):
    fiber = 0
    if spectrograph == 1:
        fiber = 1
    elif spectrograph == 2:
        fiber = 501
    spec_path = finder.get_spec_path(plate, mjd, fiber, lite=False)
    spec_path = manager.get(spec_path)
    spec = bdspec.SpecFile(spec_path)

    exposure_data = np.empty((spec.num_exposures/2, ),
                        dtype=[('PLATE', int), ('MJD', int), ('SPECTROGRAPH', int), ('PLUG_RA', float), ('PLUG_DEC', float),
                            ('R_EXTNAME', '|S30'), ('B_EXTNAME', '|S30'), ('EXP_ORDER', int), ('RA', float), ('DEC', float),
                            ('AZ', float), ('ALT', float), ('AIRMASS', float), ('R_QUALITY', '|S15'), ('B_QUALITY', '|S15'),
                            ('TAI-BEG', dt.datetime), ('TAI-END', dt.datetime)])
    for i in range(spec.num_exposures/2):
        exp_header = spec.get_exposure_hdu(i, 'r'+str(spectrograph) ).read_header()
        b_exp_header = spec.get_exposure_hdu(i, 'b'+str(spectrograph) ).read_header()
        exposure_data[i] = (int(plate), int(mjd), int(spectrograph), float(spec.header['PLUG_RA']), float(spec.header['PLUG_DEC']),
                        exp_header['EXTNAME'], b_exp_header['EXTNAME'], i, exp_header['RA'], exp_header['DEC'],
                        exp_header['AZ'], exp_header['ALT'], exp_header['AIRMASS'],
                        exp_header['QUALITY'], b_exp_header['QUALITY'],
                        tai_base_date_time + dt.timedelta(seconds=exp_header['TAI-BEG']),
                        tai_base_date_time + dt.timedelta(seconds=exp_header['TAI-END']))
    if not gather:
        for name in exposure_data.dtype.names:
            print name, "\t",
        print ""
        for row in exposure_data:
            for el in row:
                print el, "\t",
            print ""
        return None
    else:
        return exposure_data

def main():
    if len(sys.argv) == 3:
        file_deets(plate=sys.argv[1], mjd=sys.argv[2])
    if len(sys.argv) == 4:
        file_deets(plate=int(sys.argv[1]), mjd=int(sys.argv[2]), fiber=int(sys.argv[3]))
    if len(sys.argv) == 2:
        plates_table = Table.read(sys.argv[1], format='ascii')

        exposure_table_list = []
        exposure_table = None

        for row in plates_table:
            exposure_data = file_deets(row['PLATE'], row['MJD'], 1, gather=True)
            if exposure_data is not None:
                exposure_table_list.append(Table(exposure_data))
            exposure_data = file_deets(row['PLATE'], row['MJD'], 2, gather=True)
            if exposure_data is not None:
                exposure_table_list.append(Table(exposure_data))

        if len(exposure_table_list):
            if len(exposure_table_list) > 1:
                exposure_table = vstack(exposure_table_list)
            else:
                exposure_table = exposure_table_list[0]
            exposure_table.write("exposure_metadata.dat", format="ascii")

if __name__ == '__main__':
    main()

import numpy as np
from astropy.table import Table
import sys
import datetime as dt
import bossdata.path as bdpath
import bossdata.remote as bdremote
import bossdata.spec as bdspec
import bossdata.plate as bdplate
from speclite import accumulate
from speclite import resample
import matplotlib.pyplot as plt
import posixpath

#from pydl.pydlspec2d.spec2d import combine1fiber

from progressbar import ProgressBar, Percentage, Bar

'''
This script takes in a list of plate-mjd-fiber combos as output by e.g. bossquery for
bossquery --what "PLATE,MJD,FIBER" \
--where "OBJTYPE='SKY' and THING_ID=-1 and ZWARNING in (1,5) and PLATE between 4501 and 6000" \
--sort "PLATE,MJD,FIBER" --full --max-rows 10000000 --save sky_fibers_4501_to_6000.dat

It groups entries in this list by plate-mjd, then stacks the spectra for all fibers
in each group, per exposure (so you end up with 7 sky spectra for the 79 fibers in the
plate-mjd 3586-55181.)

Since it works with exposures and no co-adds, it must process the data a bit in order to be
able to stack it:  namely, resample to a regular wavelength spacing (0.975 A) over a standard
range (3500.26 to 10422.76), combining the red and blue cameras.

Each of these combined sky specta is then saved out as a CSV file named:
    stacked_sky_{plate}-{mjd}-exp{exposure:02d}.csv
'''

finder = bdpath.Finder()
manager = bdremote.Manager()

skyexp_wlen_start = 3500.26
skyexp_wlen_delta = 0.975
skyexp_wlen_out = np.arange(skyexp_wlen_start, 10403, skyexp_wlen_delta)

skyexp_loglam_start = 3.5441
skyexp_loglam_delta = 0.00006675
skyexp_loglam_out = np.arange(skyexp_loglam_start, 4.01716, skyexp_loglam_delta)

def get_stacked_fiducial_wlen_pixel_offset(target):
    return np.rint((target - skyexp_wlen_start)/skyexp_wlen_delta)

def camera_id(camera, fiber):
    return camera[0] + ("1" if (fiber <= 500) else "2")

def stack_exposures(fiber_group, exposure=None):
    exposure_list = []
    spec_sky_list = None

    filename = finder.get_plate_plan_path(plate=fiber_group['PLATE'][0], mjd=fiber_group['MJD'][0])
    plan = bdplate.Plan(manager.get(filename))

    if len(exposure_list) == 0:
        if exposure is None:
            # exposure_list.extend(range(plan.num_exposures/2))
            exposure_list.extend(range(plan.num_science_exposures))
        elif hasattr(exposure, '__iter__'):
            exposure_list.extend(exposure)
        else:
            exposure_list.append(exposure)
    if spec_sky_list is None:
        spec_sky_list = [None]*len(exposure_list)

    fiber_list_1 = fiber_group['FIBER'][fiber_group['FIBER'] <= 500]
    fiber_list_2 = fiber_group['FIBER'][fiber_group['FIBER'] > 500]

    def _prepend_frame_path(plate, frame_file):
        return posixpath.join(finder.redux_base, '{:04d}'.format(plate), frame_file)

    def _get_frame_data(fiber_list, camera, exp):
        use_calibrated = True

        exposure = plan.get_exposure_name(exp, camera, fiber_list[0], calibrated=use_calibrated)
        frame = bdplate.FrameFile(manager.get(_prepend_frame_path(fiber_group['PLATE'][0], exposure)),
                                    1 if fiber_list[0] <= 500 else 2, use_calibrated)
        data = frame.get_valid_data(fiber_list, include_sky=True, use_ivar=True)
        data['flux'] += data['sky']
        return data

    for i, exp in enumerate(exposure_list):
        r_1_data = _get_frame_data(fiber_list_1, 'red', exp)
        b_1_data = _get_frame_data(fiber_list_1, 'blue', exp)
        spec_sky_list[i] = resample_regular(b_1_data, r_1_data, spec_sky_list[i], use_loglam=False)

        r_2_data = _get_frame_data(fiber_list_2, 'red', exp)
        b_2_data = _get_frame_data(fiber_list_2, 'blue', exp)
        spec_sky_list[i] = resample_regular(b_2_data, r_2_data, spec_sky_list[i], use_loglam=False)

    '''
    for row in fiber_group:
        filename = finder.get_spec_path(plate=row['PLATE'], mjd=row['MJD'], fiber=row['FIBER'], lite=False)
        spec = bdspec.SpecFile(manager.get(filename))

        if len(exposure_list) == 0:
            if exposure is None:
                exposure_list.extend(range(plan.num_exposures/2))
            elif hasattr(exposure, '__iter__'):
                exposure_list.extend(exposure)
            else:
                exposure_list.append(exposure)
        if spec_sky_list is None:
            spec_sky_list = [None]*len(exposure_list)

        for i, exp in enumerate(exposure_list):
            r_data = spec.get_valid_data(include_sky=True, exposure_index=exp,
                            camera=camera_id('r', row['FIBER']), use_loglam=False, use_ivar=True,
                            include_wdisp=False)
            b_data = spec.get_valid_data(include_sky=True, exposure_index=exp,
                            camera=camera_id('b', row['FIBER']), use_loglam=False, use_ivar=True,
                            include_wdisp=False)
            spec_sky_list[i] = resample_regular(b_data, r_data, spec_sky_list[i], use_loglam=False)
    '''

    return exposure_list, spec_sky_list

def resample_regular(b_data, r_data, accumulate_result, use_loglam=False):
    '''
    Fiber exposures are all on slightly different grids, with slightly different starting
    points; and these are not the co-add fiducial grid.  E.g. a 'red' spectra might have 3150
    pixels between something like 5800 and 10400 A, while the fid. grid has only roughly 2550
    pixels in this range.

    In order to build sky spectra that use both cameras we need to be able to combine multiple
    fibers; this requires lining up wlen's that are all slightly offset from each other.
    Additionally, need to be able to merge red and blue; this would be easiest if the two
    cameras' overlapping region used common bins.

    Seems reasonable to establish a new "exposure fiducial" grid; from inspecting data, the smallest
    wavelength delta is about 0.975 A, ranging up to about 1.120 A.  Using this smaller value,
    a nice-ish round number of 7100 spans from 3500.26 to 10422.76.
    '''

    def _r_and_a(row, passthrough, accumulate_result):
        resample_data = resample(row, ('wavelength' if not use_loglam else 'loglam'),
                                        (skyexp_wlen_out if not use_loglam else skyexp_loglam_out), passthrough)
        return accumulate(accumulate_result, resample_data, data_out=accumulate_result,
                    join=('wavelength' if not use_loglam else 'loglam'),
                    add=('flux'), weight='ivar')

    passthrough = [name for name in b_data.dtype.names if name != ('wavelength' if not use_loglam else 'loglam')]
    for b_row, r_row in zip(b_data, r_data):
        accumulate_result = _r_and_a(b_row, passthrough, accumulate_result)
        accumulate_result = _r_and_a(r_row, passthrough, accumulate_result)
    return accumulate_result

def save_stacks(stacks, fiber_group, exposure, save_clean=True):
    plate = fiber_group[0]['PLATE']
    mjd = fiber_group[0]['MJD']
    for stackedexp, exp in zip(stacks, exposure):
        exp_table = Table(data=stackedexp)
        '''
        #exp_table['ivar'][np.abs(exp_table['ivar']) > 1000000] = np.iinfo(int).max
        mask |= (np.abs(exp_table['ivar']) < 0.000001)

        if save_clean:
            exp_table['flux'][mask] = 0
            exp_table['sky'][mask] = 0
            exp_table['ivar'][mask] = 0
        else:
            exp_table.add_column(Column(mask, name='mask'))
        '''
        exp_table.write("stacked_sky_{}-{}-exp{:02d}.csv".format(plate, mjd, exp), format="ascii.csv")

def main():
        #old = np.seterr(all='raise')

        sky_fibers_table = Table.read(sys.argv[1], format='ascii')
        sky_fibers_table = sky_fibers_table.group_by(["PLATE", "MJD"])

        progress_bar = ProgressBar(widgets=[Percentage(), Bar()], maxval=len(sky_fibers_table)).start()
        counter = 0

        for group in sky_fibers_table.groups:
            exposures, stacks = stack_exposures(group)
            save_stacks(stacks, group, exposures)
            counter += len(group)
            progress_bar.update(counter)
        progress_bar.finish()

if __name__ == '__main__':
    main()

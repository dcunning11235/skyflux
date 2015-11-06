
from ICAize import load_all_in_dir
import numpy as np
from astropy.table import Table

spectra_path = "."

def main():
    flux_arr, exp_arr, ivar_arr, mask_arr, wavelengths = \
        load_all_in_dir(spectra_path, use_con_flux=False, recombine_flux=False,
                        pattern="stacked*exp??????.csv")
    t = Table()

    t["wavelengths"] = wavelengths

    valid_per_wl = np.sum(~mask_arr, axis=0)
    invalid_per_wl_mask = (valid_per_wl == 0)
    t["valid_per_wl"] = valid_per_wl

    t["ivar_avg_per_wl"] = np.mean(ivar_arr, axis=0)
    t["ivar_stdev_per_wl"] = np.std(ivar_arr, axis=0)

    t["flux_flat_avg_per_wl"] = np.mean(flux_arr, axis=0)

    flux_weighted_avg_per_wl = np.zeros(wavelengths.shape)
    ivar_arr[:, invalid_per_wl_mask] = 1
    t["flux_weighted_avg_per_wl"] = np.average(flux_arr, axis=0, weights=ivar_arr)
    ivar_arr[:, invalid_per_wl_mask] = 0

    flux_stdev_per_wl = np.std(flux_arr, axis=0)
    t["flux_stdev_per_wl"] = flux_stdev_per_wl

    invalid_per_wl_mask = (flux_stdev_per_wl == 0)
    flux_stdev_per_wl[invalid_per_wl_mask] == 1
    t["flux_derived_ivar_per_wl"] = np.power(flux_stdev_per_wl, -2)
    flux_stdev_per_wl[invalid_per_wl_mask] == 0
    t["flux_derived_ivar_per_wl"][invalid_per_wl_mask] = 0

    t.write("flux_stats.csv", format="ascii.csv")

if __name__ == '__main__':
    main()

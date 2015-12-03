
from ICAize import load_all_in_dir
import numpy as np
from astropy.table import Table

spectra_path = "."

def main():
    flux_arr, exp_arr, ivar_arr, mask_arr, wavelengths = \
        load_all_in_dir(spectra_path, use_con_flux=False, recombine_flux=False,
                        pattern="stacked*exp??????.csv")
    t = Table()

    t["exp_id"] = exp_arr

    t["flux_flat_avg__blue"] = np.mean(flux_arr[:,:2700], axis=1)
    t["flux_flat_avg__red"] = np.mean(flux_arr[:,2700:], axis=1)

    t["flux_weighted_avg__blue"] = np.average(flux_arr[:,:2700], axis=1, weights=ivar_arr[:,:2700])
    t["flux_weighted_avg__red"] = np.average(flux_arr[:,2700:], axis=1, weights=ivar_arr[:,2700:])

    t["flux_median__blue"] = np.median(flux_arr[:,:2700], axis=1)
    t["flux_median__red"] = np.median(flux_arr[:,2700:], axis=1)

    t.write("flux_averages.csv", format="ascii.csv")

if __name__ == '__main__':
    main()

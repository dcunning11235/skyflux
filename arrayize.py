import pickle
import ICAize
import numpy as np

def main():
    flux_arr, exp_arr, ivar_arr, mask_arr, wavelengths = \
        ICAize.load_all_in_dir('.', use_con_flux=False, recombine_flux=False,
                        pattern="stacked*exp??????.csv")
    np.savez("compacted_flux_data.npz", flux=flux_arr, exp=exp_arr, ivar=ivar_arr, wavelengths=wavelengths)

if __name__ == '__main__':
    main()

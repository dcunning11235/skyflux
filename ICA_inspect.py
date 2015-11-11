import ICAize
import stack
import matplotlib.pyplot as plt
import numpy as np
import random_forest_spectra as rfs
import sklearn.metrics as sm
import sys
import os.path
import pickle

path = '.'

if len(sys.argv) == 2:
    path = sys.argv[1]

fastica = ICAize.unpickle_FastICA(target_type="combined")
'''
for comp_i in range(min(fastica.components_.shape[0], 10)):
    scale_factor = 2.4/np.max(np.abs(fastica.components_[comp_i]))
    plt.plot(stack.skyexp_wlen_out, (fastica.components_[comp_i]*scale_factor)+(5*comp_i) )
plt.show()
plt.close()
'''

#c_sources, c_mixing, c_exposures, c_wavelengths = rfs.load_spectra_data('.',
#						target_type='combined', use_spca=False)

loaded_pickle = False

output = None
try:
    output = open(os.path.join(path, 'combined_flux_arr.pkl'), 'rb')
    comb_flux_arr = pickle.load(output)
    loaded_pickle = True
except:
    comb_flux_arr, comb_exposure_arr, comb_ivar_arr, comb_masks, comb_wavelengths = \
            ICAize.load_all_in_dir(path, use_con_flux=False, recombine_flux=False,
                            pattern="stacked*exp??????.csv", ivar_cutoff=0)
finally:
    if output is not None:
        output.close()

if not loaded_pickle:
    output = open(os.path.join(path, 'combined_flux_arr.pkl'), 'wb')
    pickle.dump(comb_flux_arr, output)
    output.close()

transformed = fastica.transform(comb_flux_arr)
recovered_sources = fastica.inverse_transform(transformed)

#vw_explvar = sm.explained_variance_score(comb_flux_arr, recovered_sources, multioutput="variance_weighted")
uw_explvar = sm.explained_variance_score(comb_flux_arr, recovered_sources) #, multioutput="uniform_average")
#vw_r2 = sm.r2_score(comb_flux_arr, recovered_sources, multioutput="variance_weighted")
uw_r2 = sm.r2_score(comb_flux_arr, recovered_sources) #, multioutput="uniform_average")
mse = sm.mean_squared_error(comb_flux_arr, recovered_sources) #, multioutput="uniform_average")
print transformed.shape
print uw_explvar #, vw_explvar
print uw_r2 #, vw_r2
print mse

'''
fastica = ICAize.unpickle_PCA(target_type="combined")
for comp_i in range(min(fastica.components_.shape[0], 10)):
    scale_factor = 2.4/np.max(np.abs(fastica.components_[comp_i]))
    plt.plot(stack.skyexp_wlen_out, (fastica.components_[comp_i]*scale_factor)+(5*comp_i) )

plt.show()
plt.close()

fastica = ICAize.unpickle_SPCA(target_type="combined")
for comp_i in range(min(fastica.components_.shape[0], 10)):
    scale_factor = 2.4/np.max(np.abs(fastica.components_[comp_i]))
    plt.plot(stack.skyexp_wlen_out, (fastica.components_[comp_i]*scale_factor)+(5*comp_i) )

plt.show()
plt.close()
'''

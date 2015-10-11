import matplotlib.pyplot as plt
from astropy.table import Table
import numpy as np
import seaborn as sns
import pandas as pd
from scipy.stats import kendalltau

sns.set()

import fnmatch
import os
import os.path
import sys

def main():
    ann_metadata = pd.read_csv(sys.argv[1])

    if len(sys.argv) == 3:
        ann_metadata["AVG_5K_FLUX"] = 0

        path = '.'
        for file in os.listdir(path):
            if fnmatch.fnmatch(file, sys.argv[2]):
                data = Table.read(os.path.join(path, file), format="ascii.csv")
                mask = (data['ivar'] == 0)
                total_flux = data['flux']+data['con_flux']
                avg_5k_flux = np.average(total_flux[1518:1558], weights=data['ivar'][1518:1558])

                file_split = file.split("-")
                plate = file_split[0][12:]
                mjd = file_split[1]
                exp = file_split[2][3:]
                index_val = ann_metadata[(ann_metadata["PLATE"]==int(plate)) &
                            (ann_metadata["MJD"]==int(mjd)) &
                            (ann_metadata["EXP_ID"]==int(exp))].index
                ann_metadata.loc[index_val,"AVG_5K_FLUX"] = avg_5k_flux
    '''
    del ann_metadata["TAI-END"]
    del ann_metadata["TAI-BEG"]
    del ann_metadata["PLATE"]
    del ann_metadata["MJD"]
    del ann_metadata["EXP_ID"]
    '''

    hue=None
    if len(sys.argv) == 3:
        ann_metadata = ann_metadata[ann_metadata["AVG_5K_FLUX"]!=0]
        #hue="AVG_5K_FLUX"

    g = sns.pairplot(ann_metadata, vars={"LUNAR_MAGNITUDE", "LUNAR_ELV",
        "LUNAR_SEP", "SOLAR_ELV", "SOLAR_SEP"}, hue=hue)
    plt.show()
    plt.close()

    g = sns.pairplot(ann_metadata, vars={"LUNAR_MAGNITUDE", "LUNAR_ELV",
        "LUNAR_SEP"}, hue=hue)
    plt.show()
    plt.close()

    g = sns.pairplot(ann_metadata, vars={"SOLAR_ELV", "SOLAR_SEP", "SS_AREA", "SS_COUNT"})
    plt.show()
    plt.close()

    g = sns.pairplot(ann_metadata, vars={"AIRMASS", "GALACTIC_CORE_SEP",
        "GALACTIC_PLANE_SEP", "AVG_5K_FLUX"})
    plt.show()
    plt.close()

    g = sns.jointplot(ann_metadata["ALT"], ann_metadata["AIRMASS"], stat_func=kendalltau)
    plt.show()
    plt.close()

    g = sns.jointplot(ann_metadata["ALT"], ann_metadata["LUNAR_SEP"], stat_func=kendalltau)
    plt.show()
    plt.close()


if __name__ == '__main__':
    main()

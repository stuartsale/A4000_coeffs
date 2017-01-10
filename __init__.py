import numpy as np
import scipy.interpolate as si
import numpy.ma as ma
import math
import os


module_dir = os.path.dirname(os.path.abspath(__file__))


class R_curves:

    def __init__(self, bands):

        self.A1_splines = {}
        self.A2_splines = {}

        for band in bands:
            self.A1_splines[band] = []
            self.A2_splines[band] = []

        for R in np.arange(2.1, 5.6, 0.1):
            with open("{0:s}/{1:d}_curves.out".format(module_dir, int(R*10)),
                      'r') as f:
                first_line = f.readline().split()

            columns_required_keys = []
            columns_required_values = []

            columns_required_keys.append("Teff")
            columns_required_values.append(first_line.index("Teff")-1)
            for band in bands:
                columns_required_keys.append("A_{}_1".format(band))
                columns_required_values.append(
                    first_line.index("A_{}_1".format(band))-1)

                columns_required_keys.append("A_{}_2".format(band))
                columns_required_values.append(
                    first_line.index("A_{}_2".format(band))-1)

            R_file = ma.masked_invalid(np.genfromtxt("{0:s}/{1:d}_curves.out".
                                       format(module_dir, int(R*10)),
                                       usecols=columns_required_values)[::-1])

            Teff_col = columns_required_keys.index("Teff")
            Teff_data = np.log10(ma.filled(R_file[:, Teff_col],
                                 fill_value=50000))

            for band in bands:
                u_col = columns_required_keys.index("A_{}_2".format(band))
                v_col = columns_required_keys.index("A_{}_1".format(band))

                self.A2_splines[band].append(si.UnivariateSpline(
                    ma.masked_array(Teff_data, mask=R_file[:, u_col].mask).
                    compressed(), R_file[:, u_col].compressed(), k=1))

                self.A1_splines[band].append(si.UnivariateSpline(
                    ma.masked_array(Teff_data, mask=R_file[:, v_col].mask).
                    compressed(),  R_file[:, v_col].compressed(), k=1))


class R_set:

    def __init__(self, bands):
        self.bands = bands

        self.R_set = R_curves(self.bands)

    def A_X(self, band, log_Teff, A_4000, R_5495=3.1):
        R_index = int(np.rint((R_5495-2.1)/0.1))
        return (self.R_set.A2_splines[band][R_index](log_Teff) * A_4000*A_4000
                + self.R_set.A1_splines[band][R_index](log_Teff) * A_4000)

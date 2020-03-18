"""
Liam Connor 26 April 2019

FRB-benchmarking tools 
Code to check if a given FRB guess matches the true 
DM and arrival times. 
"""
from __future__ import print_function

import json
import sys
from enum import Enum
import numpy as np
import pandas
from scipy import interpolate
import matplotlib as mpl

#mpl.use("Agg", warn=False)
#from frb_eyra_analysis import simulate_frb
#from frb_eyra_analysis import tools


def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


try:
    import simpulse
except:
    eprint("simpulse package not available")


class DetectionDecision:
    """ Class to decide if an FRB has been 
    detected or not. Each method 
    compares the true pulse parameters 
    with the 'guess' and makes a decision. 
    """

    def __init__(
        self,
        dm,
        t0,
        width_i=0.001,
        spec_ind=0.0,
        freq_ref=None,
        scat_tau_ref=0.0,
        freq=(1550, 1250),
        nfreq=1536,
        fluence=1,
        dt=8.192e-5,
    ):

        self._dm = dm
        self._t0 = t0
        self._width_i = width_i
        self._spec_ind = spec_ind
        self._freq_ref = freq_ref

        if freq_ref is None:
            self._freq_ref = 0.5 * (freq[0] + freq[-1])

        self._scat_tau_ref = scat_tau_ref
        self._freqs = np.linspace(freq[0], freq[-1], nfreq)
        self._bw = self._freqs.max() - self._freqs.min()
        self._delta_freq = self._bw / nfreq
        self._nfreq = nfreq
        self._freq_hi_MHz, self._freq_lo_MHz = freq
        self._fluence = fluence
        self._dt = dt

    def gen_simpulse(self, ntime=10000):
        """ Generate pulse dynamic spectrum 
        with simpulse 
        """

        undispersed_arrival_time = 0.5 * ntime * self.dt
        #        undispersed_arrival_time -= 4148*self.dm*(self.freq_hi_MHz**-2)
        sp = simpulse.single_pulse(
            ntime,
            self._nfreq,
            self._freq_lo_MHz,
            self._freq_hi_MHz,
            self._dm,
            self._scat_tau_ref,
            self._width_i,
            self._fluence,
            self.spec_ind,
            undispersed_arrival_time,
        )

        data_simpulse = np.zeros([self._nfreq, ntime])
        sp.add_to_timestream(data_simpulse, 0.0, ntime * self._dt)
        data_simpulse = data_simpulse[::-1]

        return data_simpulse

    def gen_injfrb_pulse(
        self, ntime=10000, upchan_factor=1, upsamp_factor=1, conv_dmsmear=False
    ):
        """ Generate pulse dynamic spectrum 
        with injectfrb.simulate_frb
        """
        data_bg = np.zeros([upchan_factor * self._nfreq, upsamp_factor * ntime])
        data_injfrb, p = simulate_frb.gen_simulated_frb(
            NFREQ=upchan_factor * self._nfreq,
            NTIME=upsamp_factor * ntime,
            sim=True,
            fluence=self._fluence,
            spec_ind=self._spec_ind,
            width=self._width_i,
            dm=self._dm,
            background_noise=data_bg,
            delta_t=self._dt / upsamp_factor,
            plot_burst=False,
            freq=(self._freq_hi_MHz, self._freq_lo_MHz),
            FREQ_REF=self._freq_ref,
            scintillate=False,
            scat_tau_ref=self._scat_tau_ref,
            disp_ind=2.0,
            conv_dmsmear=conv_dmsmear,
        )

        data_injfrb = data_injfrb.reshape(
            self._nfreq, upchan_factor, ntime, upsamp_factor
        )
        data_injfrb = data_injfrb.mean(1).mean(-1)

        return data_injfrb

    def gen_dm_time_gaussian(self, dm_err=10., t_err=0.1):
#        sigdm = 5.0 + 0.025 * self._dm
#        sigt = 0.1 * (1 + self._width_i / 0.001)  # scale for pulse width, min 0.1 sec

        ntime = 1000
        ndm = 1000

        times = np.linspace(0, 100 * self._dt * ntime, ntime)
        times -= 0.5 * ntime * self._dt * 100
        times += self._t0
        dms = np.linspace(0.0 * self._dm, 2 * self._dm, ndm)

        dmtarr = np.exp(
            -0.5 * (self._t0 - times[None]) ** 2 / t_err ** 2
            - 0.5 * (self._dm - dms[:, None]) ** 2 / dm_err ** 2
        )
        dmtarr_function = interpolate.interp2d(times, dms, dmtarr)

        return dmtarr, dms, times, dmtarr_function

    def gen_dm_time_bowtie(self, simulator="simpulse"):
        if simulator is "injectfrb":
            data = self.gen_injfrb_pulse(
                ntime=10000, upchan_factor=1, upsamp_factor=1, conv_dmsmear=False
            )
        elif simulator is "simpulse":
            data = self.gen_simpulse(ntime=10000)
        else:
            eprint("Expected either (injectfrb, simpulse)")
            return

        freq_ref = 0.5 * (self._freqs[0] + self._freqs[-1])

        data_event_copy = data.copy()

        data_event_copy = tools.dedisperse(
            data_event_copy,
            self._dm,
            self._dt,
            freq=(self._freqs[0], self._freqs[-1]),
            freq_ref=freq_ref,
        )

        mm = np.argmax(data_event_copy.mean(0))
        data_event_copy = data_event_copy[:, mm - 500 : mm + 500]
        dmtarr, dms, times = tools.dm_transform(
            data_event_copy,
            self._freqs,
            dt=self._dt,
            freq_ref=freq_ref,
            dm_min=-50,
            dm_max=50,
            ndm=50,
        )
        times += self._t0
        dms += self._dm

        dmtarr /= dmtarr.max()
        dmtarr_function = interpolate.interp2d(times, dms, dmtarr)

        return dmtarr, dms, times, dmtarr_function

    def dm_time_box_decision(self, dm_guess, t0_guess, dm_err=10., t_err=0.1, width_dep=False):
        """ method to test if parameter 
        guess is within the allowed DM_time_box

        Parameters
        ----------
        dm_guess : 
            guess of FRB's dispersion measure 
        t0_guess : 
            guess of FRB's arrival time 
        dm_err : 
            allowed fractional error on DM 
        t_err : 
            allowed arrival time error in seconds 
        width_dep: bool
            width dependent box size (default False)
        """
        if width_dep is True:
            t_err = t_err * (1 + self._width_i / 0.01)

#        dm_stat = np.abs(1.0 - np.float(dm_guess) / self._dm)
        dm_stat = np.abs(self._dm - dm_guess)
        t_stat = np.abs(self._t0 - t0_guess)

        decision = (dm_stat < dm_err) & (t_stat < t_err)

        return decision


    def find_parameter_guess(self, input_df, t_err=1.0):
        """ The dm/time guess is generated by finding the 
        highest S/N event in a time window with width 2*t_err 
        around the true pulse. Current default is to have 
        no constraint from DM.
        """
        dm_arr = input_df[Column.DM].values
        t_arr = input_df[Column.time].values

        # if dm is low, use arithmetic test,
        # otherwise use geometric distance
        #if dm_err * self._dm < 50.0:
        #    dm_stat = np.abs(self._dm - dm_arr)
        #    dm_err = 50.0
        #else:
        #    dm_stat = np.abs(1.0 - dm_arr / self._dm)

        t_stat = np.abs(self._t0 - t_arr)

        ind = np.where(t_stat < t_err)[0]

        if len(ind) == 0:
            return None

        return input_df.iloc[ind][Column.SN].idxmax()

    def dm_time_contour_decision(
        self,
        dm_guess,
        t0_guess,
        thresh=0.1,
        simulator="simpulse",
        dmtarr_function="box",
        t_err=0.1,
        dm_err=10.0,
    ):

        """ Submit DM/time guess for true FRB parameters using 
        one of three dmtarr_function contours (box, gaussian, bowtie). t_err and 
        dm_err apply for the box guess. 

        Method returns: boolean decision, dmtarr, dm/time extent list 
        """

        if type(dmtarr_function) is str:
            if dmtarr_function == "bowtie":
                dmtarr, dms, times, dmtarr_function = self.gen_dm_time_bowtie(
                    simulator=simulator
                )
                extent = [times[0], times[-1], dms[-1], dms[0]]
            elif dmtarr_function == "gaussian":
                dmtarr, dms, times, dmtarr_function = self.gen_dm_time_gaussian(dm_err=dm_err, t_err=t_err)
                extent = [times[0], times[-1], dms[-1], dms[0]]
            elif dmtarr_function == "box":
                decision = self.dm_time_box_decision(
                    dm_guess, t0_guess, dm_err=dm_err, t_err=t_err
                )
                extent = [
                    t0_guess - t_err,
                    t0_guess + t_err,
                    dm_guess * (1 + dm_err),
                    dm_guess * (1 - dm_err),
                ]
                return decision, [], []

        val = dmtarr_function(t0_guess, dm_guess)
        decision = val > thresh

        return decision[0], dmtarr, extent


class Column(Enum):
    DM = 'Dispersion measure'
    SN = 'Signal to noise'
    time = 'Time of arrival (s)'
    width = 'Downsample / width'
    downfact = 'Downfact'
    width_i = 'Width_i'
    with_obs = 'Width_obs'
    spec_ind = 'Spec_ind'
    scat_tau_ref = 'Scat tau ref'
    t_samp = 'Tsamp (s)'
    bw_mhz = 'BW_MHz'
    freq_hi = 'Freq_hi'
    n_chan = 'Nchan'
    freq_ref = 'Freq_ref'
    input_index = 'Corresponding index in input'
    truth_index = 'Corresponding index in truth'


truth_columns = [
    Column.DM,
    Column.SN,
    Column.time,
    Column.width,
    Column.downfact,
    Column.width_i,
    Column.with_obs,
    Column.spec_ind,
    Column.scat_tau_ref,
    Column.t_samp,
    Column.bw_mhz,
    Column.freq_hi,
    Column.n_chan,
    Column.freq_ref,
]

input_columns = [
    Column.DM,
    Column.SN,
    Column.time,
    Column.width,
    Column.freq_ref,
]

def get_truth_box_dims(truth_df):
    """ Calculate dimensions of truth region 
    based on max width and DM of dataset. 
    Function returns a time of arrival error in 
    seconds (t_err) and a dm error in units 
    pc cm**-3"""
    width_i_arr = truth_df[Column.width_i]
    dm_arr = truth_df[Column.DM]
    freq_hi = truth_df[Column.freq_hi][0]
    bw = truth_df[Column.bw_mhz][0]
    nchan = truth_df[Column.n_chan][0]
    tsamp = truth_df[Column.t_samp][0]
    freq_low = freq_hi-bw
    freq_mid = freq_hi-bw/2.

    tdm = 8.3e-6 * dm_arr * bw / nchan * (freq_mid*1e-3)**-3
    width_obs = np.sqrt(tdm**2 + tsamp**2 + width_i_arr**2)
    
    width_obs_max = width_obs.max()
    dm_max = dm_arr.max()

    t_err = 2*width_obs_max
    dm_err = width_obs_max / (4148.0 * np.abs(freq_hi**-2-freq_low**-2))

    return t_err, dm_err

def get_truth_cand_ratio(t_err_cand, dm_err_cand, t_err_truth, dm_err_truth):
    """ Calculate ratio of truth region to candidate region, 
    to test upper limit of False Positives 
    """
    area_cand = t_err_cand*dm_err_cand
    area_truth = t_err_truth*dm_err_truth

    return area_truth/area_cand

def compare(input_df, truth_df):
    # todo: freq_ref_cand?
    freq_ref_truth = truth_df[Column.freq_ref][0]
    freq_ref_cand = input_df[Column.freq_ref][0]
    
    # time fix?
    #truth_df[Column.time] += 4148 * truth_df[Column.time] * (freq_ref_cand ** -2. - freq_ref_truth ** -2.)
    input_df[Column.time] -= 4148 * input_df[Column.DM] * (freq_ref_cand ** -2. - freq_ref_truth ** -2.)
    
    # cross reference columns
    truth_df[Column.input_index] = None
    input_df[Column.truth_index] = None

    # Calculate time and DM error for truth region
    t_err_truth, dm_err_truth = get_truth_box_dims(truth_df)
    # Use a candidate window of 1 second
    t_err_cand = 1.0
    dm_err_cand = 100 + input_df[Column.DM].max()

    for i in range(len(truth_df)):
        truth_row = truth_df.iloc[i]
        D = DetectionDecision(
            dm=truth_row[Column.DM],
            t0=truth_row[Column.time],
        )

        # highest S/N row index of input_df in box around truth dm & time
        guess_index = D.find_parameter_guess(
            input_df=input_df,
            t_err=t_err_cand,
        )

        if guess_index is None:
            continue

        # test if the guessed row counts as a 'detection'
        guess_row = input_df.iloc[guess_index]
        dec_bool, dmtarr, extent = D.dm_time_contour_decision(
            guess_row[Column.DM],
            guess_row[Column.time],
            simulator='injectfrb',
            dmtarr_function='box',
            dm_err=dm_err_truth,
            t_err=t_err_truth,
        )

        if dec_bool:
            # set references to one another
            truth_df.loc[i, Column.input_index] = guess_index
            input_df.loc[guess_index, Column.truth_index] = i


INPUT_PATH = '/data/input/implementation_output'
GROUND_TRUTH_PATH = '/data/input/ground_truth'

INPUT_PATH = '/tank/users/connor/eyra/data/input/implementation_output'
GROUND_TRUTH_PATH = '/tank/users/connor/eyra/data/input/ground_truth'

if __name__ == "__main__":
    truth_df = pandas.read_csv(GROUND_TRUTH_PATH, names=truth_columns, delim_whitespace=True, skiprows=1)
    input_df = pandas.read_csv(INPUT_PATH, names=input_columns, delim_whitespace=True)

    compare(input_df, truth_df)

    # rows from truth_df with corresponding input row
    correct_truth = truth_df[truth_df[Column.input_index].notnull()]

    # rows from input_df with corresponding truth row
    correct_input = input_df[input_df[Column.truth_index].notnull()]

    # note that multiple Input rows could map to the same Truth row!
    # e.g. the following is not per se true: n_correct == len(input_df) - n_false_positive
    n_correct = len(correct_truth)                          # truth rows detected
    n_false_negative = len(truth_df) - n_correct            # truth rows missed
    n_false_positive = len(input_df) - len(correct_input)   # false input rows

    metrics_dict = {
        'ground_truth': {
            'column_names': [f'{column.name} ({column.value})' for column in list(truth_df)],
            'data': json.loads(truth_df.to_json(orient='values'))
        },
        'implementation_output': {
            'column_names': [f'{column.name} ({column.value})' for column in list(input_df)],
            'data': json.loads(input_df.to_json(orient='values'))
        },
        # matches maps each matching input row index => truth row index.
        'matches': {input_index: truth_index for input_index, truth_index in correct_input[Column.truth_index].items()},
        'metrics': {
            '# True positives': n_correct,
            '# False positives': n_false_positive,
            '# False negatives': n_false_negative,
        }
    }

    print(json.dumps(metrics_dict))

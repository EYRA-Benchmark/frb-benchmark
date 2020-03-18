import sys

import argparse
import numpy as np
import pandas as pd 
import json
import logging
import pylab as plt
import matplotlib
from mpl_toolkits.axes_grid1 import make_axes_locatable
params = {
        'axes.labelsize' : 16,
        'font.size' : 14,
        'legend.fontsize': 14,
        'xtick.labelsize': 16,
        'ytick.labelsize': 16,
        'text.usetex': False,
        'figure.figsize': [10, 8]
        }
matplotlib.rcParams.update(params)

from blind_detection import input_columns, truth_columns, Column

def manage_input(fn):
    """ Read in output json file
    """
    with open(fn, 'r') as f:
        data = json.load(f)
    
    df_gt = pd.DataFrame(data['ground_truth']['data'], columns=data['ground_truth']['column_names'])
    df_op = pd.DataFrame(data['implementation_output']['data'], columns=data['implementation_output']['column_names'])
    
    df_op_plot = df_op[['DM (Dispersion measure)', 'SN (Signal to noise)', 'time (Time of arrival (s))']]
    df_op_plot.columns = ['out_dm', 'out_snr', 'out_toa']
    
    df_gt_plot = df_gt[['DM (Dispersion measure)', 'SN (Signal to noise)',
       'time (Time of arrival (s))','width_i (Width_i)', 'with_obs (Width_obs)',
       'spec_ind (Spec_ind)']]
    df_gt_plot.columns = ['in_dm', 'in_snr', 'in_toa', 'in_width', 'in_width_obs', 'in_si']
    
    return df_gt_plot, df_op_plot, data

def plot_arb_json(fn, param1, param2, sizeparam='snr'):
    """ Plot two parameters against one another for both 
    the ground_truth and the code output data
    """
    fig = plt.figure()

    df_gt_plot, df_op_plot, data = manage_input(fn)

    matches = data['matches']
    ind_matches = matches.values()
    
    assert param1 in ['dm', 'snr', 'toa'], "Don't recognize the first parameter"
    assert param2 in ['dm', 'snr', 'toa'], "Don't recognize the second paramater"

    data_gt_x = df_gt_plot['in_'+param1]
    data_gt_y = df_gt_plot['in_'+param2]

    data_op_x = df_op_plot['out_'+param1]
    data_op_y = df_op_plot['out_'+param2]

    size_gt = df_gt_plot['in_'+sizeparam]
    size_op = df_op_plot['out_'+sizeparam]

    plt.scatter(data_gt_x, data_gt_y, size_gt, color='k', alpha=0.5)
    plt.scatter(data_op_x[ind_matches], data_op_y[ind_matches], size_op, color='C0', alpha=1)
#    plt.scatter(data_op_x[ind_matches], data_op_y[ind_matches], size_op, color='C1', alpha=0.5)
    plt.legend(['Ground truth', 'Code output'])
    plt.xlabel(param1, fontsize=18)
    plt.ylabel(param2, fontsize=18)
    plt.show()

def plot_arb_txt(files, fntruth, param1, param2, sizeparam='snr'):
    fig = plt.figure()

    truth_df = pd.read_csv(fntruth, names=truth_columns, delim_whitespace=True, skiprows=1)
    freq_ref_truth = truth_df[Column.freq_ref][0]

    legend_str = []

    if param1 in ('time', 'toa', 'Time'):
        x = Column.time
    elif param1 in ('dm', 'DM'):
        x = Column.DM
    elif param1 in ('Width', 'width'):
        x = Column.width

    if param2 in ('time', 'toa', 'Time'):
        y = Column.time
    elif param2 in ('dm', 'DM'):
        y = Column.DM
    elif param2 in ('Width', 'width'):
        y = Column.width

    if sizeparam in ('snr', 'sig', 'SNR', 'SN'):
        msize = Column.SN
    elif sizeparam in ('dm', 'DM'):
        msize = Column.DM
    elif sizeparam in ('Width', 'width'):
        msize = Column.width

    for fn in files:
        input_df = pd.read_csv(fn, names=input_columns, delim_whitespace=True, skiprows=1)
        freq_ref_cand = input_df[Column.freq_ref][0]
        input_df[Column.time] -= 4148 * input_df[Column.DM] * (freq_ref_cand ** -2. - freq_ref_truth ** -2.)

        print(x,y)
        plt.scatter(input_df[x], input_df[y], input_df[msize], alpha=0.5)
        legend_str.append(fn.split('/')[-1])

    plt.scatter(truth_df[x], truth_df[y], truth_df[msize], alpha=0.55, color='k')
    legend_str.append('Truth')

    plt.legend(legend_str)
    plt.ylabel('DM (pc cm**-3)', fontsize=16)
    plt.xlabel('Time (s)', fontsize=16)
    plt.show()

if __name__=='__main__':
    parser = argparse.ArgumentParser(description="Generates plots to visualise output data",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-f', '--file', help='json or txt file(s)', type=str, nargs='+', required=True)
    parser.add_argument('-truth_file', '--truth_file', help='must be a .txt file', type=str, required=False)
    parser.add_argument('-json', '--json', help='json files as opposed to standard output', action='store_true')
    parser.add_argument('-param1', '--param1', help='y-axis parameter (snr, toa, dm, width)', default='toa')
    parser.add_argument('-param2', '--param2', help='x-axis parameter (snr, toa, dm, width)', default='dm')
    parser.add_argument('-d', '--display_plots', help='Display plots', action='store_true')

    inputs = parser.parse_args()

    #assert len(sys.argv)>2, "Expecting param1 param2 [filename]\nIf no file name is given, assuming data/output"

    if not inputs.display_plots:
        logging.info('Not displaying plots.')

    if inputs.json:
        plot_arb_json(inputs.file, param1, param2, sizeparam='snr')
    else:
        plot_arb_txt(inputs.file, inputs.truth_file, inputs.param1, inputs.param2)        
    

# if __name__ == '__main__':
#     parser = argparse.ArgumentParser(description="Generates plots to visualise search software performance",
#                                      formatter_class=argparse.ArgumentDefaultsHelpFormatter)
#     parser.add_argument('-f', '--file', help='json file', type=str, required=True)
#     parser.add_argument('-ss', '--snr_snr_plot', help='Save snr snr plot', action='store_true')
#     parser.add_argument('-r', '--recall_plot', help='Save 1D recall plot', action='store_true')
#     inputs = parser.parse_args()
    
#     title = os.path.splitext(inputs.file)[0].split('_')[-1]
#     df_gt_plot, df_op_plot, gt_indices, op_indices = manage_input(inputs.file)
    
#     if inputs.snr_snr_plot:
#         snr_snr_plot(df_gt_plot, df_op_plot, gt_indices, op_indices, ['dm', 'width', 'toa'], title = None, save=True)    
    
#     if inputs.recall_plot:
#         recall_1d(df_gt_plot, gt_indices, 'dm', recall_bins = 10, hist_bins = 30, title=title, save=True)



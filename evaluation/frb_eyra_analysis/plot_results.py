import sys

import numpy as np
import matplotlib.pylab as plt
import pandas as pd 
import json

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

def plot_arb(fn, param1, param2, sizeparam='snr'):
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

if __name__=='__main__':
    assert len(sys.argv)>2, "Expecting param1 param2 [filename]\nIf no file name is given, assuming data/output"

    param1 = sys.argv[1]
    param2 = sys.argv[2]

    try:
        fn = sys.argv[3]
    except:
        fn = 'data/output'
        
    plot_arb(fn, param1, param2, sizeparam='snr')




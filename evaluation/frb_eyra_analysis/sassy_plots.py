import pandas as pd
import json, argparse, os, logging
import pylab as plt
import numpy as np
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
logging_format = '%(asctime)s - %(funcName)s -%(name)s - %(levelname)s - %(message)s'
logging.basicConfig(level=logging.INFO, format=logging_format)


def histedges_equalN(x, nbin):
    """
    Generates 1D histogram with equal
    number of examples in each bin
    :param x: input data
    :param nbin: number of bins
    :return: bin edges
    """
    npt = len(x)
    return np.interp(np.linspace(0, npt, nbin + 1),
                     np.arange(npt),
                     np.sort(x))


def snr_scatter(param, ax, fig, df):
    """
    Scatter plot of Recovered S/N vs Injected
    S/N with colorscale as FRB property/parameter
    :param param: parameter to use for colorscale
    :param ax: axis object
    :param fig: figure object
    :param df: dataframe with data 
    :return: plot object
    """
    
    sc = ax.scatter(x=df['in_snr'], y=df['out_snr'], c=df['in_'+param], cmap='viridis', alpha=0.2)
    ax.plot(df['in_snr'], df['in_snr'])
    ax.set_ylabel('Recovered S/N')
    ax.grid()
    return sc


def snr_snr_plot(df_gt, df_op, gt_indices, op_indices, params, title = None, save=False, show=True):
    """
    Generates snr scatter plot for required parameters 
    in the data
    :param df_gt: dataframe with ground truth info
    :param df_op: dataframe with detected candidate info
    :param gt_indices: Ground truth indexes of candidates 
                       that were detected by the search soft
    :param op_indices: Output indexes of the candidates 
                       reported by the search soft
    :param params: list of parameters to plot
    :param title: Title of the plot
    :param save: To save the figure
    """
    
    df_scatter_plot = pd.concat([df_op.iloc[op_indices].reset_index(drop=True), 
                df_gt.iloc[gt_indices].reset_index(drop=True)], axis=1)    
    sc = []
    
    fig, ax = plt.subplots(len(params), 1, sharex=True)
    if len(params) == 1:
        ax = [ax]
    for i in range(len(params)):
        sc.append(snr_scatter(params[i], ax=ax[i], fig=fig, df=df_scatter_plot))
        divider = make_axes_locatable(ax[i])
        cax = divider.append_axes('right', size='5%', pad=0.1)
        cbar = fig.colorbar(sc[i], cax=cax)
        if 'dm' in params[i]:
            label = 'DM'
        elif 'width' in params[i]:
            label = 'Width (s)'
        elif 'snr' in params[i]:
            label = 'S/N'
        else:
            label = params[i]
        cbar.set_label(label, rotation=270)

    ax[len(params)-1].set_xlabel('Injected S/N')
    if title:
        plt.suptitle(f'{title}', y=0.93)
    fig.subplots_adjust(right=0.8, hspace=0.1)
    
    if save:
        figname = title+'_snr_snr_plot' if title else 'snr_snr_plot' 
        plt.savefig(f'{figname}.png', bbox_inches='tight')
    if show:
        plt.show()
        

def recall_1d(df_gt, gt_indices, param, figobj=None, 
              recall_bins = 100, hist_bins = 500, 
              title=None, save=False, show=True, plot_truth=True,
              sigthresh=0.0):
    """
    Generates the 1D recall plot with equal number 
    of examples in each bin, overlayed with the 
    ground truth histogram, for a given parameter
    :param df_gt: dataframe with ground truth info
    :param gt_indices: Ground truth indexes of candidates 
                       that were detected by the search soft
    :param param: parameter to plot
    :param recall_bins: number of bins for recall plot
    :param hist_bins: number of bins for param histogram
    :param title: Title of the plot
    :param save: To save the figure
    :returns: axis object of the plot 
    """
    
    param_label = ({'snr':'S/N',
                    'dm':'DM (pc cm**-3)',
                    'width':'Width (ms)'})[param]

    df_out = df_gt.iloc[gt_indices]

    if 'dm' in param or 'width' in param:
        indsnr_out = np.where(df_out['in_snr']>sigthresh)[0]
        indsnr_gt = np.where(df_gt['in_snr']>sigthresh)[0]
        df_out = df_out.iloc[indsnr_out]
        df_gt = df_gt.iloc[indsnr_gt]

    if 'snr' in param or 'width' in param:
        bins = histedges_equalN(np.log10(df_gt[f'in_{param}']), recall_bins)
        gt_hist, _ = np.histogram(np.log10(df_gt[f'in_{param}']), bins)
        out_hist, _ = np.histogram(np.log10(df_out[f'in_{param}']), bins)        
    else:
        bins = histedges_equalN(df_gt[f'in_{param}'], recall_bins)
        gt_hist, _ = np.histogram(df_gt[f'in_{param}'], bins)
        out_hist, _ = np.histogram(df_out[f'in_{param}'], bins)        
    
    recall = out_hist/gt_hist
    
    bin_mid = (bins[:-1] + bins[1:])/2
    
    if figobj is None:
        fig, ax1 = plt.subplots()
    else:
        fig, ax1 = figobj

    if plot_truth:
        ax2 = ax1.twinx()
        ax2.tick_params(axis='y')
        ax1.set_xlabel(f'{param_label}')
        ax1.set_ylabel('Recall')
        ax1.tick_params(axis='y')
        ax1.grid()

    ax1.step(bin_mid, recall)

    if 'snr' in param or 'width' in param:
        if plot_truth:
            ax2.hist(np.log10(df_gt[f'in_{param}']), alpha=0.25, bins=hist_bins)
            ax2.set_ylabel('Number simulated')
    else:
        if plot_truth:
            ax2.hist(df_gt[f'in_{param}'], alpha=0.25, bins=hist_bins)
            ax2.set_ylabel('Number simulated')

#    if 'snr' in param or 'width' in param:
#        ax1.set_xscale('log')

    if title:
        pass
#        plt.suptitle(f'{title}', y=1.01)
    fig.tight_layout()    
        
    return ax1


def manage_input(file):
    """
    Reads the json files generates by EYRA
    Benchmark scripts and generates ground
    truth and output dataframes with nice column
    names
    :param file: filepath of the json file to read
    :return df_gt: dataframe with ground truth info
    :return df_op: dataframe with detected candidate info
    :return gt_indices: Ground truth indexes of candidates 
                       that were detected by the search soft
    :return op_indices: Output indexes of the candidates 
                       reported by the search soft
    """
    with open(file, 'r') as f:
        data = json.load(f)
    
    dfgt = pd.DataFrame(data['ground_truth']['data'], columns=data['ground_truth']['column_names'])
    dfop = pd.DataFrame(data['implementation_output']['data'], columns=data['implementation_output']['column_names'])
    
    df_op = dfop[['DM (Dispersion measure)', 'SN (Signal to noise)', 'time (Time of arrival (s))']]
    df_op.columns = ['out_dm', 'out_snr', 'out_toa']
    
    df_gt = dfgt[['DM (Dispersion measure)', 'SN (Signal to noise)',
       'time (Time of arrival (s))','width_i (Width_i)', 'with_obs (Width_obs)',
       'spec_ind (Spec_ind)']]
    df_gt.columns = ['in_dm', 'in_snr', 'in_toa', 'in_width', 'in_width_obs', 'in_si']
    
    op_match_indices = []
    gt_match_indices = []
    for out_index in data['matches']:
        op_match_indices.append(int(out_index))
        gt_match_indices.append(data['matches'][out_index])
        
    return df_gt, df_op, gt_match_indices, op_match_indices

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generates plots to visualise search software performance",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-f', '--file', help='json file(s)', type=str, nargs='+', required=True)
    parser.add_argument('-ss', '--snr_snr_plot', help='Save snr snr plot', action='store_true')
    parser.add_argument('-r', '--recall_plot', help='Save 1D recall plot', action='store_true')
    parser.add_argument('-d', '--display_plots', help='Display plots', action='store_true')
    parser.add_argument('-sc', '--sig_cut', help='Use only events with true snr>sig_cut', 
                        type=float, default=0)
    parser.add_argument('-p', '--params', help='Parameter for 1D recall plot (dm, width, toa)', type=str, nargs='+', 
                        default=['dm'])
    
    inputs = parser.parse_args()
    inputs.file.sort()

    if not inputs.display_plots:
        logging.info('Not displaying plots.')
    
    legend_str=[]
    for ii,file in enumerate(inputs.file):
        algo_name = os.path.splitext(file)[0].split('/')[-1]
        legend_str.append(algo_name)
        
        logging.info(f'Reading and cleaning data from {file} for plotting.')
        df_gt_plot, df_op_plot, gt_match_indices, op_match_indices = manage_input(file)

        if inputs.snr_snr_plot:
            logging.info(f'Generating SNR-SNR plot for {algo_name}.')
            snr_snr_plot(df_gt_plot, df_op_plot, 
                         gt_match_indices, op_match_indices, ['dm', 'width', 'toa'], 
                         title=algo_name, save=True, show=inputs.display_plots)

    for param in inputs.params:
        legend_str=[]
        for ii,file in enumerate(inputs.file):
            df_gt_plot, df_op_plot, gt_match_indices, op_match_indices = manage_input(file)
            algo_name = os.path.splitext(file)[0].split('/')[-1]
            legend_str.append(algo_name)

            if inputs.recall_plot:
                if ii==0:
                    figobj = plt.subplots()
                    plot_truth = True
                else:
                    plot_truth = False
                    
                logging.info(f'Generating 1D recall plot for {algo_name}.')
                ax1 = recall_1d(df_gt_plot, gt_match_indices, param, 
                                  figobj=figobj, recall_bins=25, 
                                  hist_bins=60, title=algo_name, save=False, 
                                  show=False, 
                                  plot_truth=plot_truth, 
                                  sigthresh=inputs.sig_cut)

        ax1.legend(legend_str)

        if inputs.display_plots:
            plt.show()

        figname = f'{algo_name}_1d_recall_{param}' if algo_name else f'1d_recall_{param}' 
        plt.savefig(f'{figname}.png', bbox_inches='tight')

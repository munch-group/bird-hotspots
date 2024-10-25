
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D
from matplotlib import gridspec

data_path = Path('/project/Birds/faststorage/data')
results_path = Path('/project/Birds/faststorage/people/kmt/bird-hotspots/results')
figures_path = Path('/project/Birds/faststorage/people/kmt/bird-hotspots/figures')


base_counts = ['nA', 'nG', 'nT', 'nC']

substitution_counts =  ['nA2C', 'nA2G', 'nA2T', 'nC2A', 'nC2G', 'nC2T',
                        'nG2A', 'nG2C', 'nG2T', 'nT2A', 'nT2C', 'nT2G']
cpg_substitution_counts =  ['nA2c', 'nA2g', 'nc2A', 'nc2g', 'nc2T',
                            'ng2A', 'ng2c', 'ng2T', 'nT2c', 'nT2g']
# in paired order
substitutions = ['rA2C', 'rT2G',
                 'rC2T', 'rG2A',
                 'rA2G', 'rT2C',
                 'rA2T', 'rT2A', 
                 'rC2A', 'rG2T',
                 'rC2G', 'rG2C']
# cpg_substitutions = ['rA2c', 'rT2g',
#                      'rc2T', 'rg2A',
#                      'rA2g', 'rT2c',
#                      'rc2A', 'rg2T',
#                      'rc2g', 'rg2c']


transitions = ['rA2G', 'rG2A', 'rT2C', 'rC2T']
# cpg_transitions = ['rA2g', 'rg2A', 'rT2c', 'rc2T']

transversions = [x for x in substitutions if x not in transitions]
# cpg_transversions = [x for x in cpg_substitutions if x not in cpg_transitions]

paired_patterns = [('rT2G', 'rA2C'),
                   ('rA2G', 'rT2C'),
                   ('rA2T', 'rT2A'), 
                   ('rG2T', 'rC2A'),
                   ('rC2G', 'rG2C'), 
                   ('rC2T', 'rG2A')]
# cpg_paired_patterns = [('rT2g', 'rA2c'),
#                        ('rA2g', 'rT2c'),
#                        ('rg2T', 'rc2A'),
#                        ('rc2g', 'rg2c'), 
#                        ('rc2T', 'rg2A')]

chromosomes = ['1', '1A', '2', '3', '4', '4A', '5', '6', '7', '9', '10', '11', '12', '13', '14', '15']

example_species = ['Zebra Finch', 'Medium Ground-finch', 'Peregrine Falcon', 'American Flamingo', 'Downy Woodpecker', 'Ostrich']

flank_start = 30000

def abline(slope, intercept, ax=None):
    "Add a straight line through the plot"
    if ax is None:
        ax = plt.gca()
    x_vals = np.array(ax.get_xlim())
    y_vals = intercept + slope * x_vals
    ax.plot(x_vals, y_vals, '--', color='grey')
    
def add_lowess(x, y, ax=None, color=None, is_sorted=True, frac=0.005, it=0, **kwargs):
    "Add a lowess curve to the plot"
    if ax is None:
        ax = plt.gca() 
    filtered = lowess(y, x, is_sorted=is_sorted, frac=frac, it=it, **kwargs)
    ax.plot(filtered[:,0], filtered[:,1])

def add_band(x_low, x_high, ax=None, color='gray', linewidth=0, alpha=0.5, zorder=0, **kwargs):
    "Plot a gray block on x interval"
    if ax is None:
        ax = plt.gca()
    y_low, y_high = ax.get_ylim()
    g = ax.add_patch(Rectangle((x_low, y_low), x_high-x_low, y_high-y_low, 
                 facecolor=color,
                 linewidth=linewidth,
                 alpha=alpha,
                 zorder=zorder))

def stairs(df, start='start', end='end', pos='pos', endtrim=0):
    "Turn a df with start, end into one with pos to plot as stairs"
    df1 = df.copy(deep=True)
    df2 = df.copy(deep=True)
    df1[pos] = df1[start]
    df2[pos] = df2[end] - endtrim
    return pd.concat([df1, df2]).sort_values([start, end])

def optimize_data_frame(df, inplace=False, down_int='integer'):
    # down_int can be 'unsigned'
    
    if inplace:
        converted_df = df
    else:
        converted_df = pd.DataFrame()

    floats_optim = (df
                    .select_dtypes(include=['float'])
                    .apply(pd.to_numeric,downcast='float')
                   )
    converted_df[floats_optim.columns] = floats_optim

    ints_optim = (df
                    .select_dtypes(include=['int'])
                    .apply(pd.to_numeric,downcast=down_int)
                   )
    converted_df[ints_optim.columns] = ints_optim

    for col in df.select_dtypes(include=['object']).columns:
        num_unique_values = len(df[col].unique())
        num_total_values = len(df[col])
        if num_unique_values / num_total_values < 0.5:
            converted_df[col] = df[col].astype('category')
        else:
            converted_df[col] = df[col]

    unchanged_cols = df.columns[~df.columns.isin(converted_df.columns)]
    converted_df[unchanged_cols] = df[unchanged_cols]

    # keep columns order
    converted_df = converted_df[df.columns]      
            
    return converted_df
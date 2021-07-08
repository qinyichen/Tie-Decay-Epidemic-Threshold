import os
import time
import math
import numpy as np
import pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from tie_decay_epidemics import *


def dataframe_to_dict(edgelist):
    """Turn pandas dataframe to dict for faster query of data.

    Note: the conversion is costly.
    TODO: is there a better way to parse the data?

    Parameters
    ----------
    edgelist : dataframe
         The series of interactions between nodes/agents. The edgelist should
         have the format of [src, dst, time].

    Returns
    -------
    update_dict : dict
         The dict keys are timestamps at which the tie strengths are updated.
         The dict values are lists including the interactions to be updated at
         that time.
    """
    start_time = time.time()
    update_dict = {}
    for index, row in edgelist.iterrows():
        try:
            update_dict[int(row['time'])+1].append((row['src'], row['dst'], row['time']))
        except KeyError:
            update_dict[int(row['time'])+1] = [(row['src'], row['dst'], row['time'])]
    print ("It takes {}s to turn dataframe to dict.".format(time.time()-start_time))
    return update_dict


def plot(SIS, prefix, dirpath="../plots"):
    """Plot the SIS process w.r.t. time after the SIS process is done.

    Parameters
    ----------
    SIS : TieDecay_SIS
         A SIS process on the tie-decay network that has been performed.
    prefix : str
         Name of the plot.
    dirpath: str
         Directory where the plot is saved.
    """
    plt.figure()
    plt.plot(range(SIS.time+1), SIS.susceptible_history, label="Susceptible")
    plt.plot(range(SIS.time+1), SIS.infected_history, label="Infected")
    # plt.plot(range(SIS.time+1), SIS.critical_values, label="Critical Value")
    # plt.plot(range(SIS.time+1), np.ones(SIS.time+1), label="Threshold")
    plt.title('{}\nrateSI={}, rateIR={}, Outbreak Size={}' \
                .format(prefix, SIS.rateSI, SIS.rateIS, SIS.get_outbreak_size()))
    plt.xlabel('Time Step', fontsize=14)
    plt.ylabel('Number of Susceptible/Infected Individuals', fontsize=14)
    plt.legend()
    plt.savefig("{}/{}.png".format(dirpath, prefix))
    plt.close()


def plot_critical_value_history(SIS, fig_name, dirpath="../plots"):
    """Plot the evolution of the critical values (i.e., how dominant eigenvalue
       of the system matrix changes over time).

    Parameters
    ----------
    SIS : TieDecay_SIS
         A SIS process on the tie-decay network that has been performed.
    prefix : str
         Name of the plot.
    dirpath: str
         Directory where the plot is saved.
    """
    len = min(SIS.system_matrix_period, SIS.time)
    plt.figure()
    plt.plot(range(len), SIS.critical_values, label="critical value", color="orange")
    plt.plot(range(len), np.ones(len), linestyle='--', label="threshold", color="r")
    plt.xlabel('length of period', fontsize=14)
    plt.ylabel('critical value', fontsize=14)
    plt.legend()
    plt.savefig("{}/{}-critical-value-history-paper.png".format(dirpath, fig_name))
    plt.close()


def plot_outbreak_size(fig_name, outbreak_size, params_SI, params_IS, vmax=100):
    """Plot the outbreak sizes that correspond to different rates of infection
       and rates of recovery.

    Parameters
    ----------
    fig_name : str
         Name of the plot.
    outbreak_size : 2darray
         The matrix of outbreak sizes.
    params_SI : 1darray
         The array of rates of infection.
    params_IS : 1darray
         The array of rates of recovery.
    """
    plt.figure()
    fig, ax = plt.subplots(figsize=(15,10))
    outbreak_size_df = pd.DataFrame(outbreak_size,
                                    index=np.round(params_SI, 2),
                                    columns=np.round(params_IS, 2))
    ax = sns.heatmap(outbreak_size_df,
                     vmin=0,
                     vmax=vmax,
                     cmap='Blues',
                     annot=True)
    plt.xlabel('rate of recovery', fontsize=16)
    plt.ylabel('rate of infection', fontsize=16)
    plt.savefig('../plots/{}-outbreak-size.png'.format(fig_name), bbox_inches="tight")
    plt.close()


def plot_critical_value(fig_name, critical_value, params_SI, params_IS):
    """Plot the critical values that correspond to different rates of infection
       and rates of recovery.

    Parameters
    ----------
    fig_name : str
         Name of the plot.
    critical_value : 2darray
         The matrix of critical values.
    params_SI : 1darray
         The array of rates of infection.
    params_IS : 1darray
         The array of rates of recovery.
    """
    plt.figure()
    fig, ax = plt.subplots(figsize=(15,10))
    critical_value_df = pd.DataFrame(critical_value,
                                     index=np.round(params_SI, 2),
                                     columns=np.round(params_IS, 2))
    ax = sns.heatmap(critical_value_df,
                     vmin=0,
                     vmax=5,
                     cmap='Greens',
                     annot=False)
    plt.xlabel('rate of recovery', fontsize=16)
    plt.ylabel('rate of infection', fontsize=16)
    plt.savefig('../plots/{}-critical-value.png'.format(fig_name), bbox_inches="tight")
    plt.close()

def plot_scatter(fig_name, outbreak_size, critical_value):
    """Plot a scatter plot of final outbreak sizes vs critcal values.

    Parameters
    ----------
    fig_name : str
         Name of the plot.
    outbreak_size : 2darray
         The matrix of outbreak sizes.
    critical_value : 2darray
         The matrix of critical values.
    """
    plt.figure()
    plt.scatter(outbreak_size, critical_value)
    plt.xlabel('outbreak size', fontsize=16)
    plt.ylabel('critical value', fontsize=16)
    plt.hlines(1.0, 0, 100, linestyle="dashed", color="maroon", linewidth=2)
    plt.savefig('../plots/{}-scatter-plot.png'.format(fig_name), bbox_inches="tight")
    plt.close()

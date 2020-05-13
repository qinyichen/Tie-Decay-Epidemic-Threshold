from tie_decay_epidemics import *
from utils import *
from math import *

import pandas as pd
import numpy as np
import networkx as nx
import seaborn as sns
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from numpy import linalg as LA

import time
import IPython

"""
Try different rateSI & rateIS values to observe the changes in the outbreak
sizes and compare them against the critical value in the threshold condition.
Both rateSI and rateIS range from 0.01 to 1. Each time we start from a randomly
selected node as the initial infected node, and each pair of rateSI/rateIS
values are experimented on 10 times; the results are then averaged.

Inputs
------
ER-1: an Erdos-Renyi graph with 100 nodes and mean degree 5. There are 1000
time steps in total. WTD is an exponential distribution with scale = 10.

"""

params_SI = np.arange(0.01, 1.01, 0.01)
params_IS = np.arange(0.01, 1.01, 0.01)

print ("Running experienments with params_SI = \n{}, \nparams_IS = \n{}"\
                                     .format(params_SI, params_IS))

num_rounds = 10
num_nodes = 100
max_time = 1000

edgelist_withTime = pd.read_csv('../data/ER-1-withTime.csv')
edgelist_withTime = dataframe_to_dict(edgelist_withTime)

edgelist_withoutTime = open("../data/ER-1-withoutTime.csv", "r")
next(edgelist_withoutTime, None)  # skip the first line in the input file
init_G = nx.parse_edgelist(edgelist_withoutTime,
                      delimiter=',',
                      create_using=nx.DiGraph(),
                      nodetype=int)
init_adj = nx.adjacency_matrix(init_G)

nodes = np.array(range(num_nodes))

outbreak_size = np.zeros((len(params_SI), len(params_IS)))
critical_value = np.zeros((len(params_SI), len(params_IS)))


# Perform 10 experiments for each pair of rateSI/rateIS values
for x in range(len(params_SI)):
    for y in range(len(params_IS)):

        outbreak_size_sum = 0
        critical_value_sum = 0

        for round in range(num_rounds):
            if round == 0:
                print ("Experiment: rateSI={}, rateIS={}"\
                                .format(params_SI[x], params_IS[y]))

            infected = np.random.choice(nodes, 1)
            SIS = TieDecay_SIS(nodes, infected, edgelist_withTime,
                               rateSI=params_SI[x],
                               rateIS=params_IS[y],
                               alpha=0.01,
                               init_adj=init_adj,
                               have_system_matrix=True,
                               system_matrix_period=100,
                               verbose=False)
            SIS.run(max_time=max_time)

            outbreak_size_sum += SIS.get_outbreak_size()
            critical_value_sum += SIS.critical_values[-1]

        outbreak_size[x, y] = outbreak_size_sum/num_rounds
        critical_value[x, y] = critical_value_sum/num_rounds
        print ("Outbreak size = {}, critical_value = {}"\
                .format(outbreak_size[x, y], critical_value[x, y]))

IPython.embed()

# plt.figure()
# outbreak_size = outbreak_size + 1;
# fig, ax = plt.subplots(figsize=(15,10))
# log_norm = LogNorm(vmin=outbreak_size.min().min(), vmax=outbreak_size.max().max())
# cbar_ticks = [pow(10, i) for i in range(floor(log10(outbreak_size.min().min())), \
#                                             ceil(log10(outbreak_size.max().max())))]
# outbreak_size_df = pd.DataFrame(outbreak_size, index=params_SI, columns=params_IS)
# ax = sns.heatmap(outbreak_size_df, norm=log_norm,cbar_kws={"ticks": cbar_ticks}, \
#                      annot=False, annot_kws={"size": 16}, cmap='Blues', fmt='g')
# plt.xlabel('rateIS', fontsize=16)
# plt.ylabel('rateSI_max', fontsize=16)
# plt.title('Outbreak Size, Erdos-Renyi graph with 1000 nodes, tie-decay SIS (fast)', fontsize=16)
# plt.savefig('OS-fast.png',bbox_inches="tight")
# plt.close()
#
# plt.figure()
# fig, ax = plt.subplots(figsize=(15,10))
# critical_value_df = pd.DataFrame(critical_value, index=params_SI, columns=params_IS)
# ax = sns.heatmap(critical_value_df, vmin=0, vmax=5, cmap='Blues')
# plt.xlabel('rateIS', fontsize=16)
# plt.ylabel('rateSI_max', fontsize=16)
# plt.title('Critical Value, Erdos-Renyi graph with 1000 nodes, tie-decay SIS (slow)', fontsize=16)
# plt.savefig('critical-value-slow.png',bbox_inches="tight")
# plt.close()
#
# plt.figure()
# OS_Std = OS_Std+1;  # In order to take the logarithm
# fig, ax = plt.subplots(figsize=(15,10))
# log_norm = LogNorm(vmin=OS_Std.min().min(), vmax=OS_Std.max().max())
# cbar_ticks = [pow(10, i) for i in range(floor(log10(OS_Std.min().min())), \
#                                         ceil(log10(OS_Std.max().max())))]
# OS_Std_df = pd.DataFrame(OS_Std, index=params_SI, columns=params_IS)
# ax = sns.heatmap(OS_Std_df, norm=log_norm,cbar_kws={"ticks": cbar_ticks}, \
#                      annot=False, annot_kws={"size": 16}, cmap='Blues', fmt='g')
# plt.xlabel('rateIS', fontsize=16)
# plt.ylabel('rateSI_max', fontsize=16)
# plt.title('Outbreak Size, Erdos-Renyi graph with 1000 nodes, standard SIS (t_max = 100)', fontsize=16)
# plt.savefig('OS-std-fast.png',bbox_inches="tight")
# plt.close()
#
# plt.figure()
# fig, ax = plt.subplots(figsize=(15,10))
# threshold_Std_df = pd.DataFrame(threshold_Std, index=params_SI, columns=params_IS)
# ax = sns.heatmap(threshold_Std_df, vmin=0, vmax=12, cmap='Blues')
# plt.xlabel('rateIS', fontsize=16)
# plt.ylabel('rateSI_max', fontsize=16)
# plt.title('Critical Value, Erdos-Renyi graph with 1000 nodes, standard SIS (t_max = 100)', fontsize=16)
# plt.savefig('critical-value-std-fast.png',bbox_inches="tight")
# plt.close()

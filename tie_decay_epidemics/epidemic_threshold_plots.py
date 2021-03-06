from tie_decay_epidemics import *
from utils import *

from math import *
import numpy as np
import networkx as nx
import pandas as pd
from numpy import linalg as LA
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.ticker import FormatStrFormatter

import time
import IPython

"""
### Section 4.1 ###

Try different rateSI & rateIS values to observe the changes in the outbreak
sizes and compare them against the critical value in the threshold condition.
Both rateSI and rateIS range from 0.01 to 1. Each time we start from a randomly
selected node as the initial infected node, and each pair of rateSI/rateIS
values are experimented on 10 times; the results are then averaged.

Inputs
------
ER-1: an Erdos-Renyi graph with 100 nodes and mean degree 10. There are 1000
time steps in total. WTD is an exponential distribution with scale = 100.

ER-2: an Erdos-Renyi graph with 100 nodes and mean degree 5. There are 1000
time steps in total. WTD is an exponential distribution with scale = 100.

ER-3: an Erdos-Renyi graph with 100 nodes and mean degree 2. There are 1000
time steps in total. WTD is an exponential distribution with scale = 100.

"""

graph_idx = 1
graph_scale = 100
decay_coeff = 1e-1

fig_name = "ER-{}-{}-{}".format(graph_idx, graph_scale, decay_coeff)

params_SI = np.arange(0.05, 1.01, 0.05)
params_IS = np.arange(0.05, 1.01, 0.05)

print ("Running experienments with params_SI = \n{}, \nparams_IS = \n{}"\
                                               .format(params_SI, params_IS))
print ("Experiment on Erdos-Renyi-{}, with alpha={}, scale = {}."\
                                .format(graph_idx, decay_coeff, graph_scale))

# Read in edgelist and associated timestamps.

edgelist_withTime = pd.read_csv('../data/ER-{}-{}-withTime.csv'.format(graph_idx, graph_scale))
edgelist_withTime = dataframe_to_dict(edgelist_withTime)

edgelist_withoutTime = open("../data/ER-{}-withoutTime.csv".format(graph_idx), "r")
next(edgelist_withoutTime, None)  # skip the first line in the input file
init_G = nx.parse_edgelist(edgelist_withoutTime,
                      delimiter=',',
                      create_using=nx.DiGraph(),
                      nodetype=int)
init_adj = nx.adjacency_matrix(init_G) * 0.5
nodes = np.array(init_G.nodes)

# Initialize variables.
num_rounds = 10
max_time = 1000
outbreak_size = np.zeros((len(params_SI), len(params_IS)))
critical_value = np.zeros((len(params_SI), len(params_IS)))

start_time = time.time()

# Perform *num_rounds* experiments for each pair of rateSI/rateIS values
for x in range(len(params_SI)):
    for y in range(len(params_IS)):

        outbreak_size_sum = 0
        critical_value_sum = 0

        for round in range(num_rounds):
            if round == 0:
                print ("Experiment: rateSI={}, rateIS={}".format(params_SI[x], params_IS[y]))

            infected = np.random.choice(nodes, 1)
            SIS = TieDecay_SIS(nodes, infected, edgelist_withTime,
                               rateSI=params_SI[x],
                               rateIS=params_IS[y],
                               alpha=decay_coeff,
                               init_adj=init_adj,
                               have_system_matrix=True,
                               system_matrix_period=100,
                               have_critical_value_history=False,
                               verbose=False)
            SIS.run(max_time=max_time)

            outbreak_size_sum += SIS.get_outbreak_size()
            critical_value_sum += SIS.critical_value

        outbreak_size[x, y] = outbreak_size_sum/num_rounds
        critical_value[x, y] = critical_value_sum/num_rounds
        print ("Outbreak size = {}, critical_value = {}"\
                .format(outbreak_size[x, y], critical_value[x, y]))

print ("Experiment on Erdos-Renyi-{}, with alpha={}, scale = {}."\
                                .format(graph_idx, decay_coeff, graph_scale))
print ("Time used: {}".format(time.time()-start_time))

# IPython.embed()

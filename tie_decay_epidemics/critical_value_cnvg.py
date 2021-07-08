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
### Section 4.3 ###

Observe the evolution of critical value as we vary T on a tie-decay network
with some rate of infection and recovery.

"""

graph_idx = 2
graph_scale = 100
decay_coeff = 1e-1

fig_name = "ER-{}-{}-{}".format(graph_idx, graph_scale, decay_coeff)

params_SI = 0.30
params_IS = 0.70

print ("Running experienments with params_SI = {}, params_IS = {}"\
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

# Start the simulation process.
start_time = time.time()
max_time = 1000
infected = np.random.choice(nodes, 1)
SIS = TieDecay_SIS(nodes, infected, edgelist_withTime,
                   rateSI=params_SI,
                   rateIS=params_IS,
                   alpha=decay_coeff,
                   init_adj=init_adj,
                   have_system_matrix=True,
                   system_matrix_period=1000,
                   have_critical_value_history=True,
                   verbose=False)
SIS.run(max_time=max_time)

print ("Experiment on Erdos-Renyi-{}, with alpha={}, scale = {}."\
                                .format(graph_idx, decay_coeff, graph_scale))
print ("Time used: {}".format(time.time()-start_time))
print ("Outbreak size = {}, critical_value = {}"\
                    .format(SIS.get_outbreak_size(), SIS.critical_values[-1]))

# IPython.embed()

import pandas as pd
import numpy as np
import networkx as nx
import time
import IPython

from tie_decay_epidemics import *
from utils import *

"""
Section 4.4: Experiments on HyperText 2009 Conference data set.
"""

data = pd.read_csv("../real_world_data/ht09_contact_list.txt", delimiter="\t",\
                    header=None)
data.columns = ["time", "src", "dst"]
data["time"] = data["time"]/200

src = set(data["src"].tolist())
dst = set(data["dst"].tolist())

nodes = src.union(dst)
nodes = np.array(list(nodes))

edgelist = dataframe_to_dict(data)

# create a random initial network
init_G = nx.erdos_renyi_graph(len(nodes), 0.1) # ER network is by default undirected
init_adj = nx.adjacency_matrix(init_G) * 0.5

##### Simulation of SIS process #####
max_time = np.ceil(max(data["time"]))
print ("Total running time is {}".format(max_time))

# Initialize variables.
params_SI = np.arange(0.05, 1.01, 0.05)
params_IS = np.arange(0.05, 1.01, 0.05)

num_rounds = 10
outbreak_size = np.zeros((len(params_SI), len(params_IS)))
critical_value = np.zeros((len(params_SI), len(params_IS)))

start_time = time.time()

for x in range(len(params_SI)):
    for y in range(len(params_IS)):

        outbreak_size_sum = 0
        critical_value_sum = 0

        for round in range(num_rounds):
            if round == 0:
                print ("Experiment: rateSI={}, rateIS={}".format(params_SI[x], params_IS[y]))

            infected = np.random.choice(nodes, 1)
            SIS = TieDecay_SIS(nodes, infected, edgelist,
                               rateSI=params_SI[x],
                               rateIS=params_IS[y],
                               alpha=1e-2,
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

print ("Time used: {}".format(time.time()-start_time))

IPython.embed()

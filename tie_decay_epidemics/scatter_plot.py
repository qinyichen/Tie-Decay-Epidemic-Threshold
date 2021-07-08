import numpy as np
from scipy import stats
from utils import *

data_names = ["traditional-ER-1-100-1e1-10", "traditional-ER-1-100-1e1-100"]
pearson = dict()

for data_name in data_names:
    outbreak_size = np.load("../cache/{}-outbreak-size.npy".format(data_name))
    critical_value = np.load("../cache/{}-critical-value.npy".format(data_name))
    pearson[data_name] = stats.pearsonr(outbreak_size.flatten(), critical_value.flatten())
    print ("Pearson coefficient of {} = {}".format(data_name, pearson[data_name]))
    plot_scatter(data_name, outbreak_size, critical_value)

import networkx as nx
import numpy as np
import csv


def create_ER_graph(graph_name, N, p, t_max, scale):
    """Create an Erdos-Renyi graph G(N, p) with an exponential waiting time
    distribution. The graph has N nodes; edges are present with probability p.
    Each edge comes with a series of interactions at time from 0 to t_max.

    Parameters
    ----------
    graph_name : str
         The name of the output csv file.
    N : int
         Number of nodes in the Erdos-Renyi graph.
    p : float
         Edges of the Erdos-Renyi graph are present with probability p.
    t_max : float
         The maximum timestamp of interactions.
    scale : float
         scale parameter of the exponential distribution
    """
    G = nx.erdos_renyi_graph(N, p)

    # Take the largest connected component
    Gc = max(nx.connected_component_subgraphs(G), key=len)

    # Make the graph directed
    Gc = Gc.to_directed()

    print ("Number of nodes in largest connected component is", len(Gc.nodes))

    # For each edge, assign activities with exponential distribution
    # Activity at edge (i, j) simulates an interaction initiated by agent i
    # towards agent j. This increase the tie strength between i and j.
    with open("../data/{}-withTime.csv".format(graph_name), "w") as f:
        writer = csv.writer(f)
        writer.writerow(['src', 'dst', 'time'])
        for edge in Gc.edges:
            t = 0
            while t < t_max:
                t += np.random.exponential(scale=scale)
                if t <= t_max:
                    writer.writerow([edge[0], edge[1], t])
                else:
                    break
    f.close()

    with open("../data/{}-withoutTime.csv".format(graph_name), "w") as f:
        writer = csv.writer(f)
        writer.writerow(['src', 'dst'])
        for edge in Gc.edges:
            writer.writerow([edge[0], edge[1]])
    f.close()


if __name__ == "__main__":

    create_ER_graph("ER-1", 100, p=0.05, t_max=1000, scale=10)

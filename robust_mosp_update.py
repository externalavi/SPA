
# Robust_MOSP_Update algorithm implementation in Python

# importing stuffs
import networkx as nx
import numpy as np

# robust_mosp_update also take greedy_sosp_update for each scenario
# it contains expected energy & altitude gain
from greedy_sosp_update import Greedy_SOSP_Update


# algorithmic parameters
SCENARIOS = 5  # number of scenarios in consideration
B_max = 100  # maximum battery
B_res = 10  # reserve battery
alpha = 0.5  # tunining factor

# defining risk parameters
lam = 0.5  # variance penalty
mu = 0.3  # wrost-case penalty
epsilon = 0.1  # chance constraint


# robust_mosp_update algorithm
# with explaination
def Robust_MOSP_Update(G):  # input parameter graph G having node n and edge probablity p

    scenario_results = []  # for each scenario there'll be a result

    # STEP 1: update scenario specific SOSP trees
    for i in range(SCENARIOS):
        # for each scenario run greedy_sosp_tree
        Energy, Parent = Greedy_SOSP_Update(G)
        scenario_results.append(Energy)  # storing energy for each

    # STEP 2: build robust combined graph
    E_robust = nx.DiGraph()
    for (u, v) in G.edges():  # iterate through each graph edges
        energies = []
        for s in scenario_results:
            # extracts the energy cost of reaching node v in scenario s
            energies.append(s.get(v, float("inf")))

        energies = np.array(energies)  # making energies into array
        mean_energy = np.mean(energies)  # mean energy
        std_energy = np.std(energies)  # energy standard deviation
        max_energy = np.max(energies)  # maximum energy

        # calculating probablity to statisfy the condition that
        # "if the drone cannot traverse any edge safely with high probablity then discard it"
        prob = np.mean(energies <= (B_max - B_res))
        # why probablity written like this ?
        # for example
        # energies = [40, 55, 120, 30, 80]
        # B_max - B_res = 100
        # result = [True, True, False, True, True]
        # python interprets True = 1, False = 0
        # then [1,1,0,1,1]
        # mean = 0.8 (80%)

        # if drone cannot traverse edge with safety with high probablity then discard it
        if prob < (1-epsilon):
            continue

        w_r = mean_energy + lam*std_energy + mu*max_energy  # robust weight
        E_robust.add_edge(u, v, weight=w_r)  # adding edges into robust weight

    # STEP 3: robust shortest path computation
    if len(E_robust.edges()) == 0:
        return None  # if no robust edges present

    source = list(G.nodes())[0]  # source

    try:
        # shortest path calculation for robust weight
        path = nx.single_source_dijkstra(E_robust, source)
    except:
        path = None

    return path


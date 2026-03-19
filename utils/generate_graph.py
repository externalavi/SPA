
# graph generator for greedy_sosp_update and robust_mosp_update
# we'll be using erdos-renyi-graph which works for random graph
# An erods-renyi graph on the vertex V is a random graph which connects each
# pair of nodes {i,j} with probablity p, independent.

# importing
import networkx as nx

# graph generator


def generate_graph(n, p):  # node sizes, edge probablity input parameters
    # generates random graph using erdos-renyi model
    G = nx.erdos_renyi_graph(n, p, directed=True)
    # edges having direction in the generated graph

    if not nx.is_strongly_connected(G):  # connectivity check
        # a strongly connected graph means
        # from every node you can reach other node using following directions
        # A -> B -> C -> A strongly connected
        # A -> B -> C not strongly connected

        # finds the largest connected component
        largest = max(nx.strongly_connected_components(G), key=len)
        # if the graph is not strongly connected then it may contain multiple smaller components
        # for example
        # component 1: {0,1,2}
        # component 2: {3}
        # returns {0,1,2}

        # creates a new graph containing only the largest connected nodes
        G = G.subgraph(largest).copy()
        # example
        # if original graph was
        # 0 -> 1
        # 1 -> 2
        # 3 -> 3
        # new graph becomes
        # 0 -> 1
        # 1 -> 2
        # node 3 is removed
    return G


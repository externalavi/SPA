
# Greedy_SOSP_Update algorithm representation in python

# importing
from utils.energy_altitude import ExpectedEnergy, AltitudeGain


# greedy_sosp_update algorithm parameters
B_max = 100  # maximum battery
B_res = 10  # reserve battery
alpha = 0.5  # tunining factor between expected energy and altitude gain


# greedy sosp energy update algorithm implementation
# we'll be using Erdos-Renyi graph model
# G(n,p)
# n : no of edges
# p : probablity of edge between nodes

def Greedy_SOSP_Update(G):  # input is G
    V = list(G.nodes())  # taking nodes of G

    # initially taking all node energy as infinity
    # Greedy_SOSP_Update is also doing a part of shortest path in consideration of energy
    # initially we take it as infinity for relaxation
    # then we compare the energies

    # E_uv = Energy[u] + ExpectedEnergy (u,v) + alpha . AltitudeGain (u,v)
    # if E_uv < Energy[v]
    # for this comparision we're taking initial v's energy as infinity
    Energy = {v: float("inf") for v in V}
    Parent = {v: None for v in V}  # initial parent of v's None
    marked = {v: 0 for v in V}  # initial marked as 0's
    Aff = []  # initiall no affected nodes

    # STEP 1: process changed edges (greedy energy)
    for v in V:
        for u in G.predecessors(v):
            # energy calculation
            E_uv = Energy[u]+ExpectedEnergy(u, v)+alpha * AltitudeGain(u, v)
            if E_uv < Energy[u] and E_uv <= B_max - B_res:  # energy and battery checking
                Energy[v] = E_uv  # total energy required to reach node v
                Parent[v] = u
                marked[v] = 1
                Aff.append(v)

    # STEP 2: propagate energy update
    while len(Aff) != 0:  # affected vertices is not empty
        N = []  # empty vector
        Aff_new = []  # new affected nodes

        for v in Aff:
            neighbors = list(G.successors(v))
            N.extend(neighbors)  # adding all neighbors of v to N

        for v in N:
            for u in G.predecessors(v):  # for each predecessors neighbor u of v
                # this line controls which nodes are allowed to propagate energy updates in step 2
                if marked[u] != 1:
                    continue
                # earlier we did
                # marked[v] = 1: this happens only when the energy of node v is updated
                # so, marked[v] = 1: node v was updated in the last iteration
                # marked[v] = 0: node v did not change
                # marked nodes are the "active frontier nodes"
                # the line marked[u]!=1 : if node u was not updated previously then skip it
                # so the algorithm only propagates energy from nodes whose energy changed
                # without this condition algo check every node every time, that lead to time and space consuming
                # it explores the nodes that actually changed
                # example
                # A -> B -> C
                # Energy[A], Energy[B] updated, Energy[C] not updated
                # now the algo check
                # for the predecessor u of v
                # if u = C
                # marked[C]!=1 so continue (not taking consideration)

                # energy calculation
                E_uv = Energy[u] + \
                    ExpectedEnergy(u, v) + alpha * AltitudeGain(u, v)
                # energy and battery checking
                if E_uv < Energy[v] and E_uv <= B_max - B_res:
                    Energy[v] = E_uv
                    Parent[v] = u
                    marked[v] = 1
                    Aff_new.append(v)

        Aff = Aff_new  # affected = changed affected
    return Energy, Parent  # returning energy and parent

"""Energy-feasible greedy SOSP update.

This module keeps the spirit of the paper's incremental SOSP_Update while
replacing the scalar edge weight W(u, v) with a stable drone cost model:
expected energy + altitude-climb penalty + uncertainty/wind penalty.
"""

from heapq import heappop, heappush
from itertools import count

import networkx as nx

from utils.energy_altitude import DroneEdgeCost


def _default_source(G):
    nodes = list(G.nodes())
    if not nodes:
        raise ValueError("G must contain at least one node")
    return nodes[0]


def _normalise_inserted_edges(G, Ins):
    if Ins is None:
        return list(G.edges())
    return list(Ins)


def _state_from_tree(G, T, source, cost_fn, battery_limit):
    energy = {v: float("inf") for v in G.nodes()}
    parent = {v: None for v in G.nodes()}
    energy[source] = 0.0

    if T is None:
        return energy, parent

    if isinstance(T, dict):
        tree_energy = T.get("Energy", T.get("energy", {}))
        tree_parent = T.get("Parent", T.get("parent", {}))
        for v in G.nodes():
            if v in tree_energy:
                energy[v] = tree_energy[v]
            if v in tree_parent:
                parent[v] = tree_parent[v]
        energy[source] = 0.0
        parent[source] = None
        return energy, parent

    if isinstance(T, nx.DiGraph):
        for v, data in T.nodes(data=True):
            if v in energy and "energy" in data:
                energy[v] = data["energy"]
            if v in parent and "parent" in data:
                parent[v] = data["parent"]

        if all(value == float("inf") for node, value in energy.items() if node != source):
            queue = [source]
            while queue:
                u = queue.pop(0)
                for v in T.successors(u):
                    if not G.has_edge(u, v):
                        continue
                    candidate = energy[u] + cost_fn(u, v)
                    if candidate <= battery_limit and candidate < energy[v]:
                        energy[v] = candidate
                        parent[v] = u
                        queue.append(v)

    energy[source] = 0.0
    parent[source] = None
    return energy, parent


def _build_tree(G, energy, parent):
    T = nx.DiGraph()
    for v in G.nodes():
        T.add_node(v, energy=energy[v], parent=parent[v])
    for v, u in parent.items():
        if u is not None and G.has_edge(u, v):
            T.add_edge(u, v, **G.get_edge_data(u, v, default={}))
    return T


def Greedy_SOSP(
    G,
    T=None,
    Ins=None,
    source=None,
    B_max=100,
    B_res=10,
    alpha=0.5,
    beta=1.0,
    scenario=None,
    return_tree=False,
):
    """Update or compute an energy-feasible SOSP tree.

    Parameters mirror the paper where possible:
    ``T`` is the existing SOSP tree and ``Ins`` is the set of inserted/changed
    directed edges. If both are omitted, the function computes a fresh tree.
    """
    if not isinstance(G, nx.DiGraph):
        G = nx.DiGraph(G)

    source = _default_source(G) if source is None else source
    if source not in G:
        raise ValueError("source must be a node in G")

    battery_limit = B_max - B_res
    if battery_limit < 0:
        raise ValueError("B_res cannot be greater than B_max")

    edge_cost_cache = {}

    def cost_fn(u, v):
        if (u, v) not in edge_cost_cache:
            edge_cost_cache[(u, v)] = DroneEdgeCost(
                u, v, G=G, scenario=scenario, alpha=alpha, beta=beta
            )
        return edge_cost_cache[(u, v)]

    energy, parent = _state_from_tree(G, T, source, cost_fn, battery_limit)
    inserted_edges = _normalise_inserted_edges(G, Ins)

    heap = []
    sequence = count()

    def relax(u, v):
        if energy[u] == float("inf"):
            return False
        candidate = energy[u] + cost_fn(u, v)
        if candidate < energy[v] and candidate <= battery_limit:
            energy[v] = candidate
            parent[v] = u
            heappush(heap, (candidate, next(sequence), v))
            return True
        return False

    for u, v in inserted_edges:
        if G.has_edge(u, v):
            relax(u, v)

    while heap:
        current_energy, _, u = heappop(heap)
        if current_energy != energy[u]:
            continue
        for v in G.successors(u):
            relax(u, v)

    if return_tree:
        return energy, parent, _build_tree(G, energy, parent)
    return energy, parent


"""
"Greedy_SOSP function" finds out an energy-feasible single-source shortest path tree (SOSP)
over a directed graph "G", using a drone specific cost model.

It returns 
"energy": a dictionary mapping each node v to the minimum estimated energy 
required to each v from the soruce. 

"parent": a dictionary mapping each node v to its predecessor u on the choosen lowest-energy
route from the source. 

So technically 
'energy[v]' is the best known cumulative cost to reach 'v' from 'source'
'parent[v]' tells about the previous node on that route.

Together 'energy' & 'parent' define a tree rooted at 'source'
1. The tree contains the best energy-feasible path from the source to every reachable nodes
2. Subject to battery feasiability check

"""

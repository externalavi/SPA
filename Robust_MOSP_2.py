"""Robust energy-feasible MOSP update for drone path routing."""

from random import Random

import networkx as nx
import numpy as np

from Greedy_SOSP_2 import Greedy_SOSP
from utils.energy_altitude import DroneEdgeCost, ExpectedEnergy, UncertaintyPenalty


def _default_source_target(G, source, target):
    nodes = list(G.nodes())
    if not nodes:
        raise ValueError("G must contain at least one node")
    if source is None:
        source = nodes[0]
    if target is None:
        target = nodes[-1]
    return source, target


def generate_wind_scenarios(G, k=5, seed=7, wind_scale=0.25):
    """Create stable scenario dictionaries with edge-level wind uncertainty."""
    rng = Random(seed)
    scenarios = []
    for scenario_id in range(k):
        edge_energy = {}
        uncertainty = {}
        for u, v in G.edges():
            base_energy = ExpectedEnergy(u, v, G=G)
            base_uncertainty = UncertaintyPenalty(u, v, G=G)
            wind_multiplier = 1.0 + rng.uniform(-wind_scale, wind_scale)
            gust_penalty = max(0.0, rng.gauss(
                base_uncertainty, base_uncertainty / 3))
            edge_energy[(u, v)] = max(0.0, base_energy * wind_multiplier)
            uncertainty[(u, v)] = gust_penalty
        scenarios.append(
            {
                "id": scenario_id,
                "edge_energy": edge_energy,
                "uncertainty": uncertainty,
            }
        )
    return scenarios


def _edge_union_from_trees(trees, fallback_edges):
    edges = set()
    for tree in trees:
        edges.update(tree.edges())
    return edges or set(fallback_edges)


def Robust_MOSP(
    G,
    S=None,
    T_list=None,
    Ins=None,
    source=None,
    target=None,
    B_max=100,
    B_res=10,
    alpha=0.5,
    beta=1.0,
    lam=0.5,
    mu=0.3,
    epsilon=0.1,
    scenarios=5,
    seed=7,
):
    """Return a robust path using scenario-specific SOSP trees.

    The robust combined graph is built from the union of scenario SOSP trees,
    matching the paper's MOSP_Update structure more closely than using all
    original graph edges directly.
    """
    if not isinstance(G, nx.DiGraph):
        G = nx.DiGraph(G)

    source, target = _default_source_target(G, source, target)
    if source not in G or target not in G:
        raise ValueError("source and target must be nodes in G")

    battery_limit = B_max - B_res
    if battery_limit < 0:
        raise ValueError("B_res cannot be greater than B_max")

    if S is None:
        S = generate_wind_scenarios(G, k=scenarios, seed=seed)

    if T_list is None:
        T_list = [None] * len(S)
    if len(T_list) != len(S):
        raise ValueError("T_list must have the same length as S")

    scenario_trees = []
    scenario_distances = []
    for scenario, tree in zip(S, T_list):
        energy, parent, updated_tree = Greedy_SOSP(
            G,
            T=tree,
            Ins=Ins,
            source=source,
            B_max=B_max,
            B_res=B_res,
            alpha=alpha,
            beta=beta,
            scenario=scenario,
            return_tree=True,
        )
        scenario_distances.append(energy)
        scenario_trees.append(updated_tree)

    candidate_edges = _edge_union_from_trees(scenario_trees, G.edges())
    robust_graph = nx.DiGraph()
    robust_graph.add_nodes_from(G.nodes())

    for u, v in candidate_edges:
        if not G.has_edge(u, v):
            continue

        edge_costs = np.array(
            [
                DroneEdgeCost(u, v, G=G, scenario=scenario,
                              alpha=alpha, beta=beta)
                for scenario in S
            ],
            dtype=float,
        )
        cumulative_costs = np.array(
            [
                scenario_distances[i].get(u, float("inf")) + edge_costs[i]
                for i in range(len(S))
            ],
            dtype=float,
        )

        feasible_probability = np.mean(cumulative_costs <= battery_limit)
        if feasible_probability < 1.0 - epsilon:
            continue

        robust_weight = (
            float(np.mean(edge_costs))
            + lam * float(np.std(edge_costs))
            + mu * float(np.max(edge_costs))
        )
        robust_graph.add_edge(
            u,
            v,
            weight=robust_weight,
            feasible_probability=float(feasible_probability),
            mean_energy=float(np.mean(edge_costs)),
            max_energy=float(np.max(edge_costs)),
        )

    if not robust_graph.has_node(source) or not robust_graph.has_node(target):
        return None
    if not nx.has_path(robust_graph, source, target):
        return None

    path = nx.shortest_path(robust_graph, source, target, weight="weight")
    robust_cost = nx.path_weight(robust_graph, path, weight="weight")

    return {
        "path": path,
        "robust_cost": robust_cost,
        "robust_graph": robust_graph,
        "scenario_trees": scenario_trees,
        "scenario_distances": scenario_distances,
    }


"""
'Robust_MOSP' uses a set of 'uncertainity(wind)' to find path from 'source' to 'target'
that is robust across the multiple possible conditions.

It returns
1. path: the selected route as a node list from 'source' to 'target'
2. robust_cost: the sum of robustness-weighted edge costs along that path
3. robust_graph: a condensed directed graph built from candidate edges with robustness statistics
4. scenario_trees: one SOSP tree per scenario
5. scenario_distances: one energy-distance dictionary per scenario


"""


"""Stable edge metric helpers for drone path-routing experiments.

The previous implementation sampled a new random value every time the same
edge was evaluated. Shortest-path relaxation requires one stable cost for an
edge during a run, so these helpers first read graph edge attributes when a
graph is supplied and otherwise fall back to deterministic pseudo-random
values derived from the edge id.
"""

from hashlib import blake2b
import random


def _stable_uniform(u, v, low, high, salt):
    key = f"{salt}:{repr(u)}->{repr(v)}".encode("utf-8")
    seed = int.from_bytes(blake2b(key, digest_size=8).digest(), "big")
    return random.Random(seed).uniform(low, high)


def ExpectedEnergy(u, v, G=None, scenario=None):
    """Return expected edge energy for edge ``u -> v``.

    If ``G[u][v]["expected_energy"]`` exists, that value is used. Scenario
    dictionaries may override it with ``edge_energy[(u, v)]``.
    """
    if scenario and "edge_energy" in scenario and (u, v) in scenario["edge_energy"]:
        return scenario["edge_energy"][(u, v)]
    if G is not None and G.has_edge(u, v):
        data = G.get_edge_data(u, v, default={})
        if "expected_energy" in data:
            return data["expected_energy"]
        if "energy" in data:
            return data["energy"]
    return _stable_uniform(u, v, 10, 40, "expected_energy")


def AltitudeGain(u, v, G=None, scenario=None):
    """Return altitude change for edge ``u -> v``.

    Positive values represent climbing. Negative values represent descent.
    """
    if scenario and "altitude_gain" in scenario and (u, v) in scenario["altitude_gain"]:
        return scenario["altitude_gain"][(u, v)]
    if G is not None and G.has_edge(u, v):
        data = G.get_edge_data(u, v, default={})
        if "altitude_gain" in data:
            return data["altitude_gain"]
        if "altitude_change" in data:
            return data["altitude_change"]
    return _stable_uniform(u, v, -2, 5, "altitude_gain")


def UncertaintyPenalty(u, v, G=None, scenario=None):
    """Return an extra edge penalty for uncertainty such as wind."""
    if scenario and "uncertainty" in scenario and (u, v) in scenario["uncertainty"]:
        return scenario["uncertainty"][(u, v)]
    if scenario and "wind_penalty" in scenario and (u, v) in scenario["wind_penalty"]:
        return scenario["wind_penalty"][(u, v)]
    if G is not None and G.has_edge(u, v):
        data = G.get_edge_data(u, v, default={})
        if "uncertainty" in data:
            return data["uncertainty"]
        if "wind_penalty" in data:
            return data["wind_penalty"]
    return _stable_uniform(u, v, 0, 8, "uncertainty")


def DroneEdgeCost(u, v, G=None, scenario=None, alpha=0.5, beta=1.0):
    """Combined non-negative drone traversal cost for shortest-path updates."""
    climb_penalty = max(0.0, AltitudeGain(u, v, G=G, scenario=scenario))
    return (
        ExpectedEnergy(u, v, G=G, scenario=scenario)
        + alpha * climb_penalty
        + beta * UncertaintyPenalty(u, v, G=G, scenario=scenario)
    )

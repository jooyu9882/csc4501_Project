"""
CSC 4501 Computer Networks
QuantumNet: Routing and Congestion in a Distributed Quantum Network

STUDENT STARTER FILE

No prior quantum-computing knowledge is required. In this project:
- QPUs are network nodes.
- Quantum links are graph edges.
- An entanglement request is treated as an end-to-end network request.
- The focus is routing, reliability, capacity, queueing, congestion, and throughput.

Complete the TODO sections. Do not rename required functions unless instructed.
"""

from __future__ import annotations

import argparse
import heapq
import json
import math
import random
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd


@dataclass
class Request:
    request_id: int
    arrival_ms: float
    src: str
    dst: str


@dataclass
class RequestResult:
    request_id: int
    policy: str
    src: str
    dst: str
    arrival_ms: float
    finish_ms: float
    latency_ms: float
    waiting_ms: float
    hop_count: int
    success: bool
    path: str


# ============================================================
# TASK 1 — Build and validate the network topology [10 points]
# ============================================================

def load_topology(path: str | Path) -> nx.Graph:
    """
    Read data/topology.json and return an undirected NetworkX graph.

    Required checks:
      1. 0 < success_prob <= 1
      2. capacity >= 1
      3. the graph is connected
    """
    # TODO 1:
    # - Read the JSON file.
    # - Add all nodes.
    # - Add all links with delay_ms, success_prob, and capacity.
    # - Validate the link values and graph connectivity.
    with open(path, "r", encoding="utf-8") as file:
        data = json.load(file)

    g = nx.Graph()
    g.add_nodes_from(data["nodes"])

    for link in data["links"]:
        p = link["success_prob"]
        capacity = link["capacity"]
        if not (0 < p <= 1):
            raise ValueError("Link success probability must be between 0 and 1")
        if capacity < 1:
            raise ValueError("Link capacity must be at least 1")
        g.add_edge(link["u"], link["v"],
                   delay_ms=link["delay_ms"],
                   success_prob=p, capacity=capacity)

    if not nx.is_connected(g):
        raise ValueError("Network topology must be connected")
    return g


def draw_topology(g: nx.Graph, output_path: str | Path) -> None:
    """Provided visualization helper. You may improve the layout if desired."""
    pos = {
        "Q1": (0.0, 1.0),
        "Q2": (2.0, 1.8),
        "Q3": (2.0, 0.2),
        "Q4": (4.2, 1.8),
        "Q5": (4.2, 0.2),
        "Q6": (6.2, 1.0),
    }

    ordered_edges = [
        ("Q1", "Q2"), ("Q1", "Q3"), ("Q2", "Q3"),
        ("Q2", "Q4"), ("Q3", "Q4"), ("Q3", "Q5"),
        ("Q4", "Q5"), ("Q4", "Q6"), ("Q5", "Q6"),
    ]
    labels = {edge: f"L{i}" for i, edge in enumerate(ordered_edges, start=1)}

    plt.figure(figsize=(8.4, 3.7))
    nx.draw_networkx_nodes(g, pos=pos, node_size=1050)
    nx.draw_networkx_edges(g, pos=pos, width=1.4)
    nx.draw_networkx_labels(g, pos=pos, font_size=10)
    nx.draw_networkx_edge_labels(
        g, pos=pos, edge_labels=labels, font_size=8, rotate=False
    )
    plt.title("Six-QPU Distributed Network Topology")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(output_path, dpi=240, bbox_inches="tight")
    plt.close()


# ============================================================
# TASK 2 — Generate network traffic [10 points]
# ============================================================

def generate_requests(
    g: nx.Graph,
    n_requests: int,
    arrival_rate_per_ms: float,
    rng: random.Random,
) -> List[Request]:
    """
    Generate requests using exponentially distributed inter-arrival times.

    Hint:
        rng.expovariate(arrival_rate_per_ms)

    Every request needs:
        request_id, arrival_ms, src, dst

    Source and destination must be different.
    """
    # TODO 2
    nodes = list(g.nodes())
    if len(nodes) < 2:
        raise ValueError("At least two nodes are required")
    if arrival_rate_per_ms <= 0:
        raise ValueError("Arrival rate must be positive")

    requests = []
    arrival = 0.0
    for request_id in range(n_requests):
        arrival += rng.expovariate(arrival_rate_per_ms)
        src, dst = rng.sample(nodes, 2)
        requests.append(Request(request_id, arrival, src, dst))
    return requests


# ============================================================
# TASK 3 — Implement routing policies [30 points]
# ============================================================

def edge_weight(
    policy: str,
    attrs: dict,
    alpha: float = 1.0,
    beta: float = 15.0,
) -> float:
    """
    Required policies:

    shortest_hop:
        Minimum number of links.

    min_latency:
        Minimum total delay.

    max_reliability:
        Maximum product of link success probabilities.
        Convert this into an additive Dijkstra edge cost.

    student_policy:
        Your own policy using at least TWO link properties.
        Explain the formula and parameters in your report.
    """
    delay = attrs["delay_ms"]
    p = attrs["success_prob"]
    capacity = attrs["capacity"]

    if policy == "shortest_hop":
        # TODO 3A
        return 1.0

    if policy == "min_latency":
        # TODO 3B
        return float(delay)

    if policy == "max_reliability":
        # TODO 3C
        # Hint: logarithms convert products into sums.
        return -math.log(p)

    if policy == "student_policy":
        # TODO 3D
        # Use at least TWO of: delay, p, capacity.
        return alpha * delay / capacity + beta * (-math.log(p))

    raise ValueError(f"Unknown policy: {policy}")


def choose_path(
    g: nx.Graph,
    src: str,
    dst: str,
    policy: str,
    alpha: float = 1.0,
    beta: float = 15.0,
) -> List[str]:
    """Select a path using Dijkstra's algorithm and edge_weight(...)."""
    # TODO 3E
    return nx.dijkstra_path(
        g, src, dst,
        weight=lambda u, v, attrs: edge_weight(policy, attrs, alpha, beta)
    )


# ============================================================
# TASK 4 — Model link capacity and queueing [15 points]
# ============================================================

class LinkState:
    """
    A link has a finite number of simultaneous service slots.

    Recommended representation:
        store one next-available time per slot in a min-heap.
    """

    def __init__(self, capacity: int):
        self.capacity = capacity

        # TODO 4A:
        # Create a min-heap with one 0.0 value per capacity slot.
        self.available_times = [0.0] * capacity

        self.busy_time = 0.0
        self.jobs = 0

    def reserve(self, ready_ms: float, service_ms: float) -> Tuple[float, float]:
        """
        Return (start_time, finish_time).

        start_time = max(ready_ms, earliest_slot_free_time)
        finish_time = start_time + service_ms
        """
        # TODO 4B:
        # Update the slot heap, busy_time, and jobs.
        earliest = heapq.heappop(self.available_times)
        start = max(ready_ms, earliest)
        finish = start + service_ms
        heapq.heappush(self.available_times, finish)
        self.busy_time += service_ms
        self.jobs += 1
        return start, finish


def canonical_edge(u: str, v: str) -> Tuple[str, str]:
    return tuple(sorted((u, v)))


def make_link_states(g: nx.Graph) -> Dict[Tuple[str, str], LinkState]:
    """Create one LinkState object for each network link."""
    # TODO 4C
    return {
        canonical_edge(u, v): LinkState(attrs["capacity"])
        for u, v, attrs in g.edges(data=True)
    }


# ============================================================
# TASK 5 — End-to-end simulation and metrics [20 points]
# ============================================================

def simulate(
    g: nx.Graph,
    requests: List[Request],
    policy: str,
    seed: int,
    alpha: float = 1.0,
    beta: float = 15.0,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    For each request:
      1. choose a route;
      2. traverse links in path order;
      3. reserve link capacity;
      4. accumulate queue waiting time;
      5. sample link success/failure;
      6. stop after the first failed link;
      7. record the result.

    A link succeeds when:
        rng.random() <= success_prob
    """
    # TODO 5A:
    # Initialize random generator and link states.

    # TODO 5B:
    # Process requests in increasing arrival-time order.

    # TODO 5C:
    # Return:
    #   request_df with per-request results
    #   link_df with policy, link, jobs, utilization
    rng = random.Random(seed)
    link_states = make_link_states(g)
    results = []

    # Each event represents a request becoming ready for its next link.
    # Processing these events in time order prevents future reservations
    # from blocking requests that reach a link earlier.
    events = []
    paths = {}
    for index, request in enumerate(sorted(requests, key=lambda r: (r.arrival_ms, r.request_id))):
        paths[index] = (request, choose_path(g, request.src, request.dst,
                                            policy, alpha, beta))
        heapq.heappush(events, (request.arrival_ms, index, 0, 0.0))

    while events:
        ready_ms, index, hop_index, waiting = heapq.heappop(events)
        request, path = paths[index]

        if hop_index == len(path) - 1:
            results.append(asdict(RequestResult(
                request_id=request.request_id, policy=policy,
                src=request.src, dst=request.dst,
                arrival_ms=request.arrival_ms, finish_ms=ready_ms,
                latency_ms=ready_ms - request.arrival_ms,
                waiting_ms=waiting, hop_count=hop_index,
                success=True, path="->".join(path)
            )))
            continue

        u, v = path[hop_index], path[hop_index + 1]
        attrs = g[u][v]
        start, finish = link_states[canonical_edge(u, v)].reserve(
            ready_ms, attrs["delay_ms"]
        )
        waiting += start - ready_ms

        if rng.random() > attrs["success_prob"]:
            results.append(asdict(RequestResult(
                request_id=request.request_id, policy=policy,
                src=request.src, dst=request.dst,
                arrival_ms=request.arrival_ms, finish_ms=finish,
                latency_ms=finish - request.arrival_ms,
                waiting_ms=waiting, hop_count=hop_index + 1,
                success=False, path="->".join(path)
            )))
        else:
            heapq.heappush(events, (finish, index, hop_index + 1, waiting))

    results.sort(key=lambda r: (r["arrival_ms"], r["request_id"]))

    request_df = pd.DataFrame(results, columns=list(RequestResult.__dataclass_fields__))
    simulation_end = max((r["finish_ms"] for r in results), default=0.0)
    link_rows = []
    for (u, v), state in link_states.items():
        utilization = (state.busy_time / (state.capacity * simulation_end)
                       if simulation_end > 0 else 0.0)
        link_rows.append({"policy": policy, "link": f"{u}-{v}",
                          "jobs": state.jobs, "utilization": utilization})
    link_df = pd.DataFrame(link_rows)
    return request_df, link_df


def summarize(request_df: pd.DataFrame, offered_load: float) -> dict:
    """
    Compute:
      policy
      offered_load
      success_rate
      avg_latency_ms
      avg_waiting_ms
      avg_hops
      throughput_req_per_s
    """
    # TODO 5D
    if request_df.empty:
        return {"policy": "", "offered_load": offered_load,
                "success_rate": 0.0, "avg_latency_ms": 0.0,
                "avg_waiting_ms": 0.0, "avg_hops": 0.0,
                "throughput_req_per_s": 0.0}

    duration_ms = request_df["finish_ms"].max()
    successes = int(request_df["success"].sum())
    return {
        "policy": request_df["policy"].iloc[0],
        "offered_load": offered_load,
        "success_rate": successes / len(request_df),
        "avg_latency_ms": request_df["latency_ms"].mean(),
        "avg_waiting_ms": request_df["waiting_ms"].mean(),
        "avg_hops": request_df["hop_count"].mean(),
        "throughput_req_per_s": (1000.0 * successes / duration_ms
                                 if duration_ms > 0 else 0.0),
    }


# ============================================================
# TASK 6 — Experimental evaluation and report [15 points]
# ============================================================

def run_experiments(
    g: nx.Graph,
    output_dir: Path,
    n_requests: int = 500,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Evaluate:
        shortest_hop
        min_latency
        max_reliability
        student_policy

    Offered loads:
        0.02, 0.05, 0.10, 0.15 requests/ms

    Important:
    At one offered-load value, ALL policies must receive the SAME
    generated request trace for a fair comparison.
    """
    # TODO 6A:
    # - Generate one request trace per load.
    # - Run all four policies on the same trace.
    # - Save per-request and per-link CSV files.
    # - Save and return summary.csv.
    output_dir.mkdir(parents=True, exist_ok=True)
    policies = ["shortest_hop", "min_latency", "max_reliability", "student_policy"]
    loads = [0.02, 0.05, 0.10, 0.15]
    summaries = []

    for load in loads:
        trace = generate_requests(g, n_requests, load, random.Random(seed))
        for policy in policies:
            request_df, link_df = simulate(g, trace, policy, seed)
            label = f"{policy}_{load:.2f}"
            request_df.to_csv(output_dir / f"requests_{label}.csv", index=False)
            link_df.to_csv(output_dir / f"links_{label}.csv", index=False)
            summaries.append(summarize(request_df, load))

    summary_df = pd.DataFrame(summaries)
    summary_df.to_csv(output_dir / "summary.csv", index=False)
    return summary_df


def plot_metric(
    summary_df: pd.DataFrame,
    metric: str,
    ylabel: str,
    path: Path,
) -> None:
    """Provided plotting helper."""
    display_names = {
        "shortest_hop": "Minimum Hop",
        "min_latency": "Minimum Latency",
        "max_reliability": "Maximum Reliability",
        "student_policy": "Student Policy",
    }

    plt.figure(figsize=(6.8, 4.4))
    for policy, group in summary_df.groupby("policy"):
        group = group.sort_values("offered_load")
        offered_load_per_s = 1000.0 * group["offered_load"]
        plt.plot(
            offered_load_per_s,
            group[metric],
            marker="o",
            linewidth=1.6,
            markersize=5,
            label=display_names.get(policy, policy),
        )

    plt.xlabel("Offered Load (requests/s)")
    plt.ylabel(ylabel)
    plt.grid(axis="y", linewidth=0.5, alpha=0.35)
    plt.legend(frameon=False, loc="upper left", bbox_to_anchor=(1.02, 1))
    plt.tight_layout()
    plt.savefig(path, dpi=240, bbox_inches="tight")
    plt.close()


def make_plots(summary_df: pd.DataFrame, output_dir: Path) -> None:
    """
    Required figures:
      1. success rate vs offered load
      2. average latency vs offered load
      3. throughput vs offered load
      4. average waiting time vs offered load
    """
    # TODO 6B:
    # Call plot_metric(...) for all four required metrics.
    plot_metric(summary_df, "success_rate", "Success Rate",
                output_dir / "success_rate_vs_load.png")
    plot_metric(summary_df, "avg_latency_ms", "Average Latency (ms)",
                output_dir / "latency_vs_load.png")
    plot_metric(summary_df, "throughput_req_per_s", "Throughput (requests/s)",
                output_dir / "throughput_vs_load.png")
    plot_metric(summary_df, "avg_waiting_ms", "Average Waiting Time (ms)",
                output_dir / "waiting_vs_load.png")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--topology", default="../data/topology.json")
    parser.add_argument("--output", default="../outputs/student_run")
    parser.add_argument("--requests", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    g = load_topology(Path(args.topology))
    draw_topology(g, output_dir / "topology.png")

    summary_df = run_experiments(
        g,
        output_dir,
        n_requests=args.requests,
        seed=args.seed,
    )
    make_plots(summary_df, output_dir)

    print(summary_df.to_string(index=False))
    print(f"\nResults saved to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()

from operator import itemgetter
from queue import PriorityQueue
from typing import Any

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

from src.plotting import plot_graph


def dijkstra_sp( G: nx.Graph, source_node="0"
) -> dict[Any, list[Any]]:
    unvisited_set = set(G.nodes())
    visited_set = set()
    shortest_paths = {}
    dist = {n: np.inf for n in G}

    dist[source_node] = 0
    shortest_paths[source_node] = [source_node]# initialize shortest paths to 0

    while unvisited_set:
        node = None
        min_dist = np.inf
        for n, d in dist.items():
            if (n in unvisited_set) and (d < min_dist):
                min_dist = d
                node = n
        unvisited_set.remove(node)
        visited_set.add(node)

        for neighbor in G.neighbors(node):
            if neighbor in visited_set:
                continue
            new_dist = min_dist + G.edges[node, neighbor]['weight'] # calculate new distance
            if new_dist < dist[neighbor]:
                dist[neighbor] = new_dist
                shortest_paths[neighbor] = shortest_paths[node] + [neighbor]

    return shortest_paths

def dijkstra_sp_with_priority_queue(
        G: nx.Graph, source_node="0"
) -> dict[Any, list[Any]]:

    #unvisited_set = set(G.nodes())
    visited_set = set()
    shortest_paths = {}
    dist = {n: np.inf for n in G}

    dist[source_node] = 0
    shortest_paths[source_node] = [source_node]# initialize shortest paths to 0

    pq = PriorityQueue()
    pq.put((dist[source_node], source_node))
    while not pq.empty():
        min_dist, node = pq.get()
        #unvisited_set.remove(node)
        visited_set.add(node)
        for neighbor in G.neighbors(node):
            if neighbor in visited_set:
                continue
            new_dist = min_dist + G.edges[node, neighbor]['weight']  # calculate new distance
            if new_dist < dist[neighbor]:
                dist[neighbor] = new_dist
                shortest_paths[neighbor] = shortest_paths[node] + [neighbor]
                pq.put((new_dist, neighbor))
    return shortest_paths


if __name__ == "__main__":
    G = nx.read_edgelist("../practicum_4/graph_1.edgelist", create_using=nx.Graph)
    plot_graph(G)
    shortest_paths = dijkstra_sp_with_priority_queue(G, source_node="0")
    test_node = "5"
    shortest_path_edges = [
        (shortest_paths[test_node][i], shortest_paths[test_node][i + 1])
        for i in range(len(shortest_paths[test_node]) - 1)
    ]
    plot_graph(G, highlighted_edges=shortest_path_edges)

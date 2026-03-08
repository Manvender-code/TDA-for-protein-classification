import networkx as nx
import numpy as np
from tqdm import tqdm


def extract_baseline_features(graphs):
    """
    Pipeline 1: Extracts traditional graph theoretic features.

    Features (~14):
    - Node-level aggregated: degree (mean,std), clustering (mean,std),
      betweenness (mean,std), closeness (mean,std)
    - Graph-level: num_nodes, num_edges, density, avg_path_length (on largest CC), diameter (largest CC), spectral_radius
    """
    print("[INFO] Extracting Baseline Graph Features...")
    feature_matrix = []

    for G in tqdm(graphs, desc="Baseline Features"):

        if G.number_of_nodes() == 0:
            feature_matrix.append([0.0] * 14)
            continue

        # Node-level aggregates
        deg = list(dict(nx.degree(G)).values())
        clust = list(nx.clustering(G).values())
        betw = list(nx.betweenness_centrality(G).values())
        close = list(nx.closeness_centrality(G).values())

        num_nodes = G.number_of_nodes()
        num_edges = G.number_of_edges()
        density = nx.density(G)

        # handle disconnected graphs by taking the largest connected component
        if nx.is_connected(G):
            avg_path = nx.average_shortest_path_length(G)
            diameter = nx.diameter(G)
        else:
            largest_cc = max(nx.connected_components(G), key=len)
            subG = G.subgraph(largest_cc)
            avg_path = nx.average_shortest_path_length(subG)
            diameter = nx.diameter(subG)

        # spectral radius (largest absolute eigenvalue) - fallback 0 on failure
        try:
            eigenvalues = nx.adjacency_spectrum(G)
            spectral_radius = float(np.max(np.abs(eigenvalues)))
        except Exception:
            spectral_radius = 0.0

        features = [
            float(np.mean(deg)), float(np.std(deg)),
            float(np.mean(clust)), float(np.std(clust)),
            float(np.mean(betw)), float(np.std(betw)),
            float(np.mean(close)), float(np.std(close)),
            float(num_nodes), float(num_edges), float(density),
            float(avg_path), float(diameter), float(spectral_radius)
        ]

        feature_matrix.append(features)

    return np.array(feature_matrix, dtype=float)
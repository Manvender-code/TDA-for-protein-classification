import networkx as nx
import numpy as np
from tqdm import tqdm


def extract_baseline_features(graphs):
    """
    Improved Baseline Graph Features (Research-grade)

    Features (~18):
    - Node-level (mean, std):
        degree, clustering, betweenness, closeness
    - Graph-level:
        num_nodes, num_edges, density
        avg_path_length, diameter, radius
        transitivity
        global_efficiency
        assortativity
        spectral_radius
    """

    print("[INFO] Extracting Improved Baseline Graph Features...")
    feature_matrix = []

    for G in tqdm(graphs, desc="Baseline Features"):

        # Handle empty graph
        if G.number_of_nodes() == 0:
            feature_matrix.append([0.0] * 18)
            continue

        # -------------------------
        # Node-level features
        # -------------------------
        deg = list(dict(nx.degree(G)).values())

        clust = list(nx.clustering(G).values())

        betw = list(nx.betweenness_centrality(G).values())
        betw = [0.0 if np.isnan(x) else x for x in betw]

        close = list(nx.closeness_centrality(G).values())
        close = [0.0 if np.isnan(x) else x for x in close]

        # -------------------------
        # Basic graph features
        # -------------------------
        num_nodes = G.number_of_nodes()
        num_edges = G.number_of_edges()
        density = nx.density(G)

        # -------------------------
        # Connected component handling
        # -------------------------
        if nx.is_connected(G):
            subG = G
        else:
            largest_cc = max(nx.connected_components(G), key=len)
            subG = G.subgraph(largest_cc)

        # Path-based features
        try:
            avg_path = nx.average_shortest_path_length(subG)
        except:
            avg_path = 0.0

        try:
            diameter = nx.diameter(subG)
        except:
            diameter = 0.0

        try:
            radius = nx.radius(subG)
        except:
            radius = 0.0

        # -------------------------
        # Higher-level graph metrics
        # -------------------------
        transitivity = nx.transitivity(G)

        try:
            global_eff = nx.global_efficiency(G)
            if np.isnan(global_eff):
                global_eff = 0.0
        except:
            global_eff = 0.0

        try:
            assortativity = nx.degree_pearson_correlation_coefficient(G)
            if np.isnan(assortativity):
                assortativity = 0.0
        except:
            assortativity = 0.0

        # -------------------------
        # Spectral feature
        # -------------------------
        try:
            eigenvalues = nx.adjacency_spectrum(G)
            spectral_radius = float(np.max(np.abs(eigenvalues)))
            if np.isnan(spectral_radius):
                spectral_radius = 0.0
        except:
            spectral_radius = 0.0

        # -------------------------
        # Final feature vector
        # -------------------------
        features = [
            # Node-level statistics
            np.mean(deg), np.std(deg),
            np.mean(clust), np.std(clust),
            np.mean(betw), np.std(betw),
            np.mean(close), np.std(close),

            # Basic graph
            num_nodes, num_edges, density,

            # Path-based
            avg_path, diameter, radius,

            # Advanced topology
            transitivity,
            global_eff,
            assortativity,

            # Spectral
            spectral_radius
        ]

        feature_matrix.append(features)

    # -------------------------
    # Final cleanup (CRITICAL)
    # -------------------------
    X = np.array(feature_matrix, dtype=float)

    # Replace NaN / Inf safely
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    return X
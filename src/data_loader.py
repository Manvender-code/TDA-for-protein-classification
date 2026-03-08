import os
import requests
import zipfile
import numpy as np
import networkx as nx
from src.config import DATA_DIR, DATASET_URL, DATASET_NAME


def download_data():
    """
    Download and extract the PROTEINS_full dataset (TUDataset style).
    """
    os.makedirs(DATA_DIR, exist_ok=True)
    zip_path = os.path.join(DATA_DIR, f"{DATASET_NAME}.zip")
    extract_path = os.path.join(DATA_DIR, DATASET_NAME)

    if os.path.exists(extract_path):
        print("Dataset already downloaded.")
        return

    print("Downloading and extracting dataset...")
    r = requests.get(DATASET_URL)
    with open(zip_path, "wb") as f:
        f.write(r.content)

    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(DATA_DIR)


def load_graphs():
    """
    Parses TUDataset raw files into NetworkX graphs with node attributes.
    Returns:
        graphs (list): List of nx.Graph objects
        labels (np.array): Binary labels (Enzyme=1, Non-Enzyme=0)
    """
    base = os.path.join(DATA_DIR, DATASET_NAME, DATASET_NAME)

    # Load raw text files (TUDataset format)
    graph_indicator = np.loadtxt(f"{base}_graph_indicator.txt", dtype=int)
    edges = np.loadtxt(f"{base}_A.txt", delimiter=",", dtype=int)
    labels = np.loadtxt(f"{base}_graph_labels.txt", dtype=int)
    node_attr = np.loadtxt(f"{base}_node_attributes.txt", delimiter=",")

    # Convert labels: in PROTEINS_full, 1 -> Enzyme, 2 -> Non-Enzyme
    binary_labels = np.where(labels == 1, 1, 0)

    graphs = []
    n_graphs = np.max(graph_indicator)

    # Build empty graphs and add nodes with attributes
    for i in range(1, n_graphs + 1):
        G = nx.Graph()
        node_idx = np.where(graph_indicator == i)[0]  # indices (0-based)
        for idx in node_idx:
            # attach attribute vector under key 'attr'
            G.add_node(int(idx), attr=node_attr[int(idx)])
        graphs.append(G)

    # Add edges (TUDataset edges are 1-based indexes)
    if edges.size > 0:
        # edges might be shape (m,2) or (2,) for single edge - handle both
        edges = np.atleast_2d(edges)
        for u, v in edges:
            u0 = int(u) - 1
            v0 = int(v) - 1
            # add edge if nodes belong to same graph (graph_indicator is 1-based)
            if graph_indicator[u0] == graph_indicator[v0]:
                g_idx = int(graph_indicator[u0] - 1)
                # map node IDs to nodes used in G (we used original indices)
                graphs[g_idx].add_edge(u0, v0)

    return graphs, binary_labels
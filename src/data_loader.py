import os
import requests
import zipfile
import numpy as np
import networkx as nx
from src.config import DATA_DIR, DATASET_URL, DATASET_NAME


def download_data():
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

    # loading the graph and converting it in networkx graph structure in an array and also its labeling(binary) as graph and binary_labels

    base = os.path.join(DATA_DIR, DATASET_NAME, DATASET_NAME)

    graph_indicator = np.loadtxt(f"{base}_graph_indicator.txt", dtype=int)
    edges = np.loadtxt(f"{base}_A.txt", delimiter=",", dtype=int)
    labels = np.loadtxt(f"{base}_graph_labels.txt", dtype=int)
    node_attr = np.loadtxt(f"{base}_node_attributes.txt", delimiter=",")

    binary_labels = np.where(labels == 1, 1, 0)

    graphs = []
    n_graphs = np.max(graph_indicator)

    for i in range(1, n_graphs + 1):
        G = nx.Graph()
        node_idx = np.where(graph_indicator == i)[0]

        for idx in node_idx:
            G.add_node(idx, attr=node_attr[idx])

        graphs.append(G)

    for u, v in edges:
        u -= 1
        v -= 1
        g_id = graph_indicator[u]
        if graph_indicator[u] == graph_indicator[v]:
            graphs[g_id - 1].add_edge(u, v)

    return graphs, binary_labels

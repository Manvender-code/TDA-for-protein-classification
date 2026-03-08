import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline, FeatureUnion
from gtda.graphs import GraphGeodesicDistance
from gtda.homology import VietorisRipsPersistence

from src.config import MAX_FILTRATION_SCALE, HOMOLOGY_DIMENSIONS


# -------------------- Graph → Adjacency --------------------

class NetworkXToAdjacency(BaseEstimator, TransformerMixin):
    """
    Convert networkx.Graph objects to adjacency matrices.
    """

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        mats = []

        for G in X:
            nodes = list(G.nodes())
            idx_map = {n: i for i, n in enumerate(nodes)}
            n = len(nodes)

            if n == 0:
                mats.append(np.zeros((0, 0)))
                continue

            A = np.zeros((n, n), dtype=float)

            for u, v in G.edges():
                i = idx_map[u]
                j = idx_map[v]
                A[i, j] = 1.0
                A[j, i] = 1.0

            mats.append(A)

        return np.array(mats, dtype=object)


# -------------------- Persistence Diagram Feature Extraction --------------------

class DiagramFeatureExtractor(BaseEstimator, TransformerMixin):
    """
    Extracts the 13 canonical features from Persistence Diagrams.
    """

    def __init__(self, max_scale=15):
        self.max_scale = max_scale

    def fit(self, X, y=None):
        return self

    def transform(self, X):

        features = []

        for diagram in X:

            if diagram is None or len(diagram) == 0:
                features.append([0.0] * 13)
                continue

            h0 = diagram[diagram[:, 2] == 0]
            h1 = diagram[diagram[:, 2] == 1]
            h2 = diagram[diagram[:, 2] == 2]

            # H0 features
            h0_lengths = np.sort(h0[:, 1] - h0[:, 0])[::-1] if len(h0) else np.array([])

            f1 = h0_lengths[1] if len(h0_lengths) > 1 else 0
            f2 = h0_lengths[2] if len(h0_lengths) > 2 else 0

            valid_h0 = h0_lengths[h0_lengths < self.max_scale] if len(h0_lengths) else np.array([])
            f3 = np.sum(valid_h0) if len(valid_h0) else 0
            f4 = np.mean(valid_h0) if len(valid_h0) else 0

            # H1 features
            h1_lengths = (h1[:, 1] - h1[:, 0]) if len(h1) else np.array([])

            f5 = np.max(h1_lengths) if len(h1_lengths) else 0
            f6 = np.mean(h1_lengths) if len(h1_lengths) else 0
            f7 = np.sum(h1_lengths > 2)
            f8 = np.sum(h1_lengths > 4)
            f9 = np.sum(h1_lengths > 6)
            f10 = np.sum(h1_lengths)

            # H2 features
            f11 = np.min(h2[:, 0]) if len(h2) else 0
            f12 = np.sum(h2[:, 1] - h2[:, 0]) if len(h2) else 0
            f13 = len(h2)

            features.append([
                f1, f2, f3, f4,
                f5, f6, f7, f8,
                f9, f10, f11, f12, f13
            ])

        return np.array(features)


# -------------------- Attribute Aggregation --------------------

class AttributeAggregator(BaseEstimator, TransformerMixin):
    """
    Aggregates node attributes (mean, std, max).
    """

    def fit(self, X, y=None):
        return self

    def transform(self, X):

        stats = []

        for G in X:
            attrs = np.array([data['attr'] for _, data in G.nodes(data=True)])

            if attrs.size == 0:
                stats.append(np.zeros(29 * 3))
                continue

            mean_attr = np.mean(attrs, axis=0)
            std_attr = np.std(attrs, axis=0)
            max_attr = np.max(attrs, axis=0)

            stats.append(np.concatenate([mean_attr, std_attr, max_attr]))

        return np.array(stats)


# -------------------- Main TDA Pipeline --------------------

class GraphToFeatures:
    """
    Combines:
    - Manual TDA features
    - Node attribute statistics
    """

    @staticmethod
    def get_pipeline():

        tda_pipeline = Pipeline([
            ("to_adj", NetworkXToAdjacency()),
            ("geodesic", GraphGeodesicDistance(method="D", n_jobs=1)),
            ("vr", VietorisRipsPersistence(
                homology_dimensions=HOMOLOGY_DIMENSIONS,
                metric="precomputed",
                n_jobs=1
            )),
            ("manual_features", DiagramFeatureExtractor(MAX_FILTRATION_SCALE))
        ])

        full_pipeline = FeatureUnion([
            ("tda", tda_pipeline),
            ("attr", AttributeAggregator())
        ])

        return full_pipeline
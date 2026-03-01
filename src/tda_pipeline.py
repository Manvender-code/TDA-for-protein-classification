import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline, FeatureUnion
from gtda.graphs import GraphGeodesicDistance
from gtda.homology import VietorisRipsPersistence
from src.features import DiagramFeatureExtractor, AttributeAggregator
from src.config import MAX_FILTRATION_SCALE, HOMOLOGY_DIMENSIONS


class NetworkXToAdjacency(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        mats = []

        for G in X:
            nodes = list(G.nodes())
            idx_map = {n: i for i, n in enumerate(nodes)}
            n = len(nodes)

            A = np.zeros((n, n), dtype=float)

            for u, v in G.edges():
                i = idx_map[u]
                j = idx_map[v]
                A[i, j] = 1.0
                A[j, i] = 1.0

            mats.append(A)

        return np.array(mats, dtype=object)


class GraphToFeatures:

    @staticmethod
    def get_pipeline():

        tda_pipeline = Pipeline([
            ("to_adj", NetworkXToAdjacency()),
            ("geodesic", GraphGeodesicDistance(method="D", n_jobs=1)),
            ("vr", VietorisRipsPersistence(
                homology_dimensions=HOMOLOGY_DIMENSIONS,
                n_jobs=1,
                metric="precomputed"
            )),
            ("diagram_features", DiagramFeatureExtractor(MAX_FILTRATION_SCALE))
        ])

        full_pipeline = FeatureUnion([
            ("tda", tda_pipeline),
            ("attr", AttributeAggregator())
        ])

        return full_pipeline

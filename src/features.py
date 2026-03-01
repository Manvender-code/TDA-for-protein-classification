import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class DiagramFeatureExtractor(BaseEstimator, TransformerMixin):

    def __init__(self, max_scale=15):
        self.max_scale = max_scale

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        features = []

        for diagram in X:

            h0 = diagram[diagram[:, 2] == 0]
            h1 = diagram[diagram[:, 2] == 1]
            h2 = diagram[diagram[:, 2] == 2]

            h0_lengths = np.sort(h0[:, 1] - h0[:, 0])[::-1]
            f1 = h0_lengths[1] if len(h0_lengths) > 1 else 0
            f2 = h0_lengths[2] if len(h0_lengths) > 2 else 0

            valid_h0 = h0_lengths[h0_lengths < self.max_scale]
            f3 = np.sum(valid_h0) if len(valid_h0) else 0
            f4 = np.mean(valid_h0) if len(valid_h0) else 0

            h1_lengths = h1[:, 1] - h1[:, 0]
            f5 = np.max(h1_lengths) if len(h1_lengths) else 0
            f6 = np.mean(h1_lengths) if len(h1_lengths) else 0
            f7 = np.sum(h1_lengths > 2)
            f8 = np.sum(h1_lengths > 4)
            f9 = np.sum(h1_lengths > 6)
            f10 = np.sum(h1_lengths)

            f11 = np.min(h2[:, 0]) if len(h2) else 0
            f12 = np.sum(h2[:, 1] - h2[:, 0]) if len(h2) else 0
            f13 = len(h2)

            features.append([
                f1, f2, f3, f4,
                f5, f6, f7, f8,
                f9, f10, f11, f12, f13
            ])

        return np.array(features)


class AttributeAggregator(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        stats = []
        for G in X:
            attrs = np.array([d["attr"] for _, d in G.nodes(data=True)])
            mean = np.mean(attrs, axis=0)
            std = np.std(attrs, axis=0)
            mx = np.max(attrs, axis=0)
            stats.append(np.concatenate([mean, std, mx]))
        return np.array(stats)

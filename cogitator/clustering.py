import abc
from typing import Tuple

import numpy as np
from sklearn.cluster import KMeans


class BaseClusterer(abc.ABC):
    @abc.abstractmethod
    def cluster(
        self, embeddings: np.ndarray, n_clusters: int, **kwargs
    ) -> Tuple[np.ndarray, np.ndarray]: ...


class KMeansClusterer(BaseClusterer):
    def cluster(
        self, embeddings: np.ndarray, n_clusters: int, **kwargs
    ) -> Tuple[np.ndarray, np.ndarray]:
        random_seed = kwargs.get("random_seed", 33)
        kmeans = KMeans(n_clusters=n_clusters, random_state=random_seed, n_init=10)
        labels = kmeans.fit_predict(embeddings)
        return labels, kmeans.cluster_centers_

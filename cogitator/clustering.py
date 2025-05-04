from abc import ABC, abstractmethod
from typing import Tuple

import numpy as np
from sklearn.cluster import KMeans


class BaseClusterer(ABC):
    @abstractmethod
    def cluster(
        self, embeddings: np.ndarray, n_clusters: int, **kwargs
    ) -> Tuple[np.ndarray, np.ndarray]: ...


class KMeansClusterer(BaseClusterer):
    def cluster(
        self, embeddings: np.ndarray, n_clusters: int, **kwargs
    ) -> Tuple[np.ndarray, np.ndarray]:
        random_seed = kwargs.get("random_seed", None) or kwargs.get("seed", None)
        n_init = kwargs.get("n_init", 10)
        kmeans = KMeans(n_clusters=n_clusters, random_state=random_seed, n_init=n_init)
        labels = kmeans.fit_predict(embeddings)
        return labels, kmeans.cluster_centers_

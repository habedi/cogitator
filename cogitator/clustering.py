"""Provides abstractions and implementations for clustering algorithms."""

from abc import ABC, abstractmethod
from typing import Any, Tuple

import numpy as np
from sklearn.cluster import KMeans


class BaseClusterer(ABC):
    """Abstract base class for clustering algorithms."""

    @abstractmethod
    def cluster(
        self, embeddings: np.ndarray, n_clusters: int, **kwargs: Any
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Clusters the given embeddings into a specified number of clusters.

        Args:
            embeddings: A NumPy array where each row is an embedding vector.
            n_clusters: The desired number of clusters.
            **kwargs: Additional keyword arguments specific to the clustering implementation.

        Returns:
            A tuple containing:
                - A NumPy array of cluster labels assigned to each embedding.
                - A NumPy array of cluster centers.
        """
        ...


class KMeansClusterer(BaseClusterer):
    """A clustering implementation using the K-Means algorithm from scikit-learn."""

    def cluster(
        self, embeddings: np.ndarray, n_clusters: int, **kwargs: Any
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Clusters embeddings using K-Means.

        Args:
            embeddings: The embeddings to cluster (shape: [n_samples, n_features]).
            n_clusters: The number of clusters to form.
            **kwargs: Additional arguments for `sklearn.cluster.KMeans`.
                Supported args include `random_seed` (or `seed`) and `n_init`.

        Returns:
            A tuple containing:
                - labels (np.ndarray): Integer labels array (shape: [n_samples,]).
                - centers (np.ndarray): Coordinates of cluster centers (shape: [n_clusters, n_features]).

        Raises:
            ValueError: If `n_clusters` is invalid or embeddings are incompatible.
        """
        random_seed = kwargs.get("random_seed") or kwargs.get("seed")
        n_init = kwargs.get("n_init", "auto")
        kmeans = KMeans(
            n_clusters=n_clusters,
            random_state=random_seed,
            n_init=n_init,
            init="k-means++",
        )
        labels = kmeans.fit_predict(embeddings)
        return labels, kmeans.cluster_centers_

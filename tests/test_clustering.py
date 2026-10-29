"""Tests for cogitator.clustering module."""

import numpy as np
import pytest

from cogitator.clustering import BaseClusterer, KMeansClusterer


class TestBaseClusterer:
    """Tests for BaseClusterer abstract class."""

    def test_abstract_class_cannot_instantiate(self):
        """Test that BaseClusterer cannot be instantiated directly."""
        with pytest.raises(TypeError):
            BaseClusterer()  # type: ignore

    def test_concrete_implementation_works(self):
        """Test that a concrete implementation can be created."""

        class ConcreteClusterer(BaseClusterer):
            def cluster(self, embeddings, n_clusters, **kwargs):
                labels = np.zeros(len(embeddings), dtype=int)
                centers = np.zeros((n_clusters, embeddings.shape[1]))
                return labels, centers

        clusterer = ConcreteClusterer()
        embeddings = np.array([[1.0, 2.0], [3.0, 4.0]])
        labels, centers = clusterer.cluster(embeddings, 2)
        assert len(labels) == 2
        assert centers.shape == (2, 2)


class TestKMeansClusterer:
    """Tests for KMeansClusterer class."""

    def test_kmeans_clustering(self):
        """Test standard KMeans clustering."""
        clusterer = KMeansClusterer()
        # Generate distinct points to cluster
        embeddings = np.array([
            [1.0, 1.0],
            [1.1, 0.9],
            [10.0, 10.0],
            [10.2, 9.8]
        ])
        
        labels, centers = clusterer.cluster(embeddings, n_clusters=2, random_seed=42)
        
        assert len(labels) == 4
        assert labels[0] == labels[1]
        assert labels[2] == labels[3]
        assert labels[0] != labels[2]
        assert centers.shape == (2, 2)

    def test_kmeans_clustering_with_seed_and_init(self):
        """Test KMeans parameters like seed (alias) and n_init."""
        clusterer = KMeansClusterer()
        embeddings = np.array([
            [1.0, 1.0],
            [2.0, 2.0],
            [3.0, 3.0]
        ])
        labels, centers = clusterer.cluster(
            embeddings,
            n_clusters=2,
            seed=42,
            n_init=10
        )
        assert len(labels) == 3
        assert centers.shape == (2, 2)

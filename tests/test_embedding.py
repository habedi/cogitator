"""Tests for cogitator.embedding module."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from cogitator.embedding import BaseEmbedder, SentenceTransformerEmbedder


class TestBaseEmbedder:
    """Tests for BaseEmbedder abstract class."""

    def test_abstract_class_cannot_instantiate(self):
        """Test that BaseEmbedder cannot be instantiated directly."""
        with pytest.raises(TypeError):
            BaseEmbedder()

    def test_concrete_implementation_works(self):
        """Test that a concrete implementation can be created."""

        class ConcreteEmbedder(BaseEmbedder):
            def encode(self, texts):
                return [np.array([1.0, 2.0, 3.0]) for _ in texts]

        embedder = ConcreteEmbedder()
        result = embedder.encode(["test"])
        assert len(result) == 1
        assert isinstance(result[0], np.ndarray)


class TestSentenceTransformerEmbedder:
    """Tests for SentenceTransformerEmbedder class."""

    @pytest.fixture(autouse=True)
    def reset_singleton(self):
        """Reset singleton state before each test."""
        # Store original state
        orig_instance = SentenceTransformerEmbedder._instance
        orig_model = SentenceTransformerEmbedder._model

        # Reset singleton
        SentenceTransformerEmbedder._instance = None
        SentenceTransformerEmbedder._model = None

        yield

        # Restore original state after test
        SentenceTransformerEmbedder._instance = orig_instance
        SentenceTransformerEmbedder._model = orig_model

    def test_singleton_pattern(self):
        """Test that singleton pattern works."""
        with patch('cogitator.embedding.SentenceTransformer') as mock_st:
            mock_model = MagicMock()
            mock_st.return_value = mock_model

            embedder1 = SentenceTransformerEmbedder()
            embedder2 = SentenceTransformerEmbedder()

            assert embedder1 is embedder2
            # Model should only be created once
            mock_st.assert_called_once()

    def test_encode_returns_list(self):
        """Test that encode returns a list of arrays."""
        with patch('cogitator.embedding.SentenceTransformer') as mock_st:
            mock_model = MagicMock()
            mock_embeddings = [np.array([1.0, 2.0]), np.array([3.0, 4.0])]
            mock_model.encode.return_value = mock_embeddings
            mock_st.return_value = mock_model

            embedder = SentenceTransformerEmbedder()
            result = embedder.encode(["text1", "text2"])

            assert len(result) == 2
            mock_model.encode.assert_called_once_with(
                ["text1", "text2"],
                convert_to_numpy=True,
                show_progress_bar=False
            )

    def test_encode_with_single_text(self):
        """Test encoding a single text."""
        with patch('cogitator.embedding.SentenceTransformer') as mock_st:
            mock_model = MagicMock()
            mock_embeddings = [np.array([1.0, 2.0, 3.0])]
            mock_model.encode.return_value = mock_embeddings
            mock_st.return_value = mock_model

            embedder = SentenceTransformerEmbedder()
            result = embedder.encode(["single text"])

            assert len(result) == 1

    def test_encode_with_empty_list(self):
        """Test encoding an empty list."""
        with patch('cogitator.embedding.SentenceTransformer') as mock_st:
            mock_model = MagicMock()
            mock_model.encode.return_value = []
            mock_st.return_value = mock_model

            embedder = SentenceTransformerEmbedder()
            result = embedder.encode([])

            assert result == []

    def test_model_not_initialized_error(self):
        """Test RuntimeError when model is not initialized."""
        with patch('cogitator.embedding.SentenceTransformer') as mock_st:
            mock_st.return_value = MagicMock()
            embedder = SentenceTransformerEmbedder()

            # Force model to None
            SentenceTransformerEmbedder._model = None

            with pytest.raises(RuntimeError, match="Embedder model not initialized"):
                embedder.encode(["test"])

    def test_default_model_name(self):
        """Test that default model is all-MiniLM-L6-v2."""
        with patch('cogitator.embedding.SentenceTransformer') as mock_st:
            mock_st.return_value = MagicMock()

            embedder = SentenceTransformerEmbedder()

            mock_st.assert_called_once_with("all-MiniLM-L6-v2")

    def test_custom_model_name(self):
        """Test initialization with custom model name."""
        with patch('cogitator.embedding.SentenceTransformer') as mock_st:
            mock_st.return_value = MagicMock()

            embedder = SentenceTransformerEmbedder(model_name="custom-model")

            mock_st.assert_called_once_with("custom-model")

    def test_inherits_from_base_embedder(self):
        """Test that SentenceTransformerEmbedder inherits from BaseEmbedder."""
        with patch('cogitator.embedding.SentenceTransformer') as mock_st:
            mock_st.return_value = MagicMock()

            embedder = SentenceTransformerEmbedder()

            assert isinstance(embedder, BaseEmbedder)

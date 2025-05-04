import argparse
import importlib
import logging
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

# Add examples directory to path to allow importing
examples_dir = Path(__file__).parent.parent / "examples"
sys.path.insert(0, str(examples_dir.parent))

# List of example script filenames
example_scripts = [f.stem for f in examples_dir.glob("run_*.py")]


@pytest.fixture
def mock_get_llm_examples(mocker):
    """ Mocks get_llm specifically for the examples """
    mock_llm_instance = MagicMock()
    mock_llm_instance.generate.return_value = "Example Mock LLM Response"
    mock_llm_instance.generate_async.return_value = "Example Mock Async LLM Response"
    # Add mocks for other methods used by examples if necessary
    # e.g., fit, fit_async, run, run_async for specific strategies
    mock_llm_instance.fit = MagicMock()
    mock_llm_instance.fit_async = MagicMock(return_value=None)  # Needs to be awaitable
    mock_llm_instance.init_pool = MagicMock()
    mock_llm_instance.init_pool_async = MagicMock(return_value=None)
    mock_llm_instance.train = MagicMock()
    mock_llm_instance.train_async = MagicMock(return_value=None)
    mock_llm_instance.run = MagicMock(return_value="Example Mock Run")
    mock_llm_instance.run_async = MagicMock(
        return_value="Example Mock Async Run")  # Needs to be awaitable
    mock_llm_instance.generate_json = MagicMock(return_value={"final_answer": "Mock JSON"})
    mock_llm_instance.generate_json_async = MagicMock(
        return_value={"final_answer": "Mock Async JSON"})  # Needs to be awaitable

    # Mock the get_llm function within the examples.shared module
    return mocker.patch("examples.shared.get_llm", return_value=mock_llm_instance)


@pytest.fixture
def mock_embedding_clustering_examples(mocker):
    """ Mocks embedding and clustering for examples that might use them """
    mocker.patch("cogitator.embedding.SentenceTransformerEmbedder.encode",
                 return_value=[MagicMock()])
    mocker.patch("cogitator.clustering.KMeansClusterer.cluster",
                 return_value=(MagicMock(), MagicMock()))


# --- Test Runner ---

@pytest.mark.parametrize("script_name", example_scripts)
def test_example_script_sync(script_name, mock_get_llm_examples, mock_embedding_clustering_examples,
                             mocker, capsys):
    """ Tests if example scripts run synchronously without errors """
    logging.info(f"Testing example (sync): {script_name}")
    # Mock sys.argv to prevent interference and set provider
    mocker.patch("sys.argv", ["examples/" + script_name + ".py", "--provider",
                              "ollama"])  # Use ollama as it requires no key

    try:
        # Import the module
        module = importlib.import_module(f"examples.{script_name}")
        # Check if main_sync exists and run it
        if hasattr(module, "main_sync"):
            module.main_sync(
                argparse.Namespace(provider="ollama", model_name="mock_model", openai_key=None,
                                   use_async=False))
            # Capture stdout to check for basic execution (optional)
            captured = capsys.readouterr()
            assert "Error" not in captured.err  # Basic check for errors in output
            logging.info(f"Sync run for {script_name} completed.")
        else:
            pytest.skip(f"No main_sync function found in {script_name}")
    except Exception as e:
        pytest.fail(f"Example script {script_name} (sync) failed: {e}")


@pytest.mark.asyncio
@pytest.mark.parametrize("script_name", example_scripts)
async def test_example_script_async(script_name, mock_get_llm_examples,
                                    mock_embedding_clustering_examples, mocker, capsys):
    """ Tests if example scripts run asynchronously without errors """
    logging.info(f"Testing example (async): {script_name}")
    # Mock sys.argv to prevent interference and set provider + async flag
    mocker.patch("sys.argv",
                 ["examples/" + script_name + ".py", "--provider", "ollama", "--use-async"])

    try:
        module = importlib.import_module(f"examples.{script_name}")
        if hasattr(module, "main_async"):
            # Mock asyncio.gather if necessary, especially for complex async flows
            # mocker.patch('asyncio.gather', return_value=["Mock Async Result"])

            # Mock the specific CoT methods used in async examples if get_llm mock isn't enough
            # Example: mocker.patch('cogitator.AutoCoT.fit_async', return_value=None)
            # Example: mocker.patch('cogitator.AutoCoT.run_async', return_value="Mock Async Output")
            # (Covered by mock_get_llm_examples for now, but might need refinement)

            await module.main_async(
                argparse.Namespace(provider="ollama", model_name="mock_model", openai_key=None,
                                   use_async=True))
            captured = capsys.readouterr()
            assert "Error" not in captured.err
            logging.info(f"Async run for {script_name} completed.")
        else:
            pytest.skip(f"No main_async function found in {script_name}")
    except Exception as e:
        pytest.fail(f"Example script {script_name} (async) failed: {e}")

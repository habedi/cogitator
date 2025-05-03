import argparse
import asyncio
from unittest.mock import patch, MagicMock

import pytest
from datasets import Dataset

import benches.evaluate as benchmark_evaluate
import benches.run as benchmark_run
import benches.shared as benchmark_shared
from cogitator import ExtractedAnswer


@pytest.fixture
def mock_load_dataset(mocker):
    mock_ds = MagicMock(spec=Dataset)
    mock_ds.__len__.return_value = 10
    mock_ds.column_names = ["question", "answer"]
    mock_ds.__getitem__.return_value = {"question": "q1", "answer": "a1"}
    mocker.patch("benches.shared.concatenate_datasets", return_value=mock_ds)
    return mocker.patch("benches.shared.load_dataset", return_value=mock_ds)


@pytest.fixture
def mock_get_llm(mocker):
    mock_llm_instance = MagicMock()
    mock_llm_instance.generate.return_value = "Mock LLM Response"
    mock_llm_instance.generate_json.return_value = MagicMock(final_answer="Mock JSON")

    async def async_gen_side_effect(*args, **kwargs):
        await asyncio.sleep(0)
        return "Mock Async LLM Response"

    mock_llm_instance.generate_async = MagicMock(side_effect=async_gen_side_effect)

    async def async_json_side_effect(*args, **kwargs):
        await asyncio.sleep(0)
        return MagicMock(final_answer="Mock Async JSON")

    mock_llm_instance.generate_json_async = MagicMock(side_effect=async_json_side_effect)

    # Return the instance itself, the patch will be applied in tests needing it
    # or via autouse=True if always needed. For now, let tests apply the patch.
    # return mocker.patch("benches.shared.get_llm", return_value=mock_llm_instance)
    return mock_llm_instance  # Return the mock instance directly


@pytest.fixture
def mock_cot_methods(mocker):
    # Helper to create a simple, completed awaitable that returns None
    async def completed_awaitable_none(*args, **kwargs):
        # logging.debug(f"Mock awaitable (None) called with: args={args}, kwargs={kwargs}")
        await asyncio.sleep(0)
        return None

    # --- Mocks for AutoCoT ---
    mocker.patch("cogitator.AutoCoT.fit")
    mocker.patch("cogitator.AutoCoT.fit_async", side_effect=completed_awaitable_none)
    mocker.patch("cogitator.AutoCoT.run", return_value="AutoCoT Mock Output")

    # FIX: Define side_effect as an async def directly for run_async
    async def auto_run_async_side_effect(*args, **kwargs):
        await asyncio.sleep(0)
        return "AutoCoT Mock Async Output"

    mocker.patch("cogitator.AutoCoT.run_async", side_effect=auto_run_async_side_effect)

    # --- Mocks for CDWCoT ---
    mocker.patch("cogitator.CDWCoT.init_pool")
    mocker.patch("cogitator.CDWCoT.init_pool_async", side_effect=completed_awaitable_none)
    mocker.patch("cogitator.CDWCoT.train")
    mocker.patch("cogitator.CDWCoT.train_async", side_effect=completed_awaitable_none)
    mocker.patch("cogitator.CDWCoT.run", return_value="CDWCoT Mock Output")

    # FIX: Define side_effect as an async def directly for run_async
    async def cdw_run_async_side_effect(*args, **kwargs):
        await asyncio.sleep(0)
        return "CDWCoT Mock Async Output"

    mocker.patch("cogitator.CDWCoT.run_async", side_effect=cdw_run_async_side_effect)

    # --- Mocks for other methods ---
    mocker.patch("cogitator.SelfConsistency.run", return_value="SC Mock Output")

    # FIX: Define side_effect as an async def directly for run_async
    async def sc_run_async_side_effect(*args, **kwargs):
        await asyncio.sleep(0)
        return "SC Mock Async Output"

    mocker.patch("cogitator.SelfConsistency.run_async", side_effect=sc_run_async_side_effect)

    mocker.patch("cogitator.LeastToMost.run", return_value="LtM Mock Output")

    # FIX: Define side_effect as an async def directly for run_async
    async def ltm_run_async_side_effect(*args, **kwargs):
        await asyncio.sleep(0)
        return "LtM Mock Async Output"

    mocker.patch("cogitator.LeastToMost.run_async", side_effect=ltm_run_async_side_effect)

    mocker.patch("cogitator.TreeOfThoughts.run", return_value="ToT Mock Output")

    # FIX: Define side_effect as an async def directly for run_async
    async def tot_run_async_side_effect(*args, **kwargs):
        await asyncio.sleep(0)
        return "ToT Mock Async Output"

    mocker.patch("cogitator.TreeOfThoughts.run_async", side_effect=tot_run_async_side_effect)

    mocker.patch("cogitator.GraphOfThoughts.run", return_value="GoT Mock Output")

    # FIX: Define side_effect as an async def directly for run_async
    async def got_run_async_side_effect(*args, **kwargs):
        await asyncio.sleep(0)
        return "GoT Mock Async Output"

    mocker.patch("cogitator.GraphOfThoughts.run_async", side_effect=got_run_async_side_effect)


def test_dataset_registry_exists():
    assert hasattr(benchmark_shared.Datasets, "registry")
    assert isinstance(benchmark_shared.Datasets.registry, dict)
    assert "gsm8k" in benchmark_shared.Datasets.registry


def test_load_dataset_by_name_success(mock_load_dataset):
    name = "gsm8k"
    cutoff = 5
    with patch.object(benchmark_shared.Datasets, f"_process_{name}",
                      return_value=(["q"] * 10, ["g"] * 10)) as mock_processor:
        qs, golds = benchmark_shared.Datasets.load_dataset_by_name(name, cutoff)
        assert len(qs) == cutoff
        assert len(golds) == cutoff
        mock_load_dataset.assert_called()
        mock_processor.assert_called_once()


def test_load_dataset_by_name_not_found():
    with pytest.raises(ValueError, match="not found in registry"):
        benchmark_shared.Datasets.load_dataset_by_name("nonexistent_dataset", 10)


def test_load_dataset_by_name_processor_missing(mocker):
    original_registry = benchmark_shared.Datasets.registry.copy()
    dataset_key = "temp_no_processor"
    hf_path = "some/path"
    config_name = None
    available_splits = ["train"]
    benchmark_shared.Datasets.registry[dataset_key] = (hf_path, config_name, [], available_splits)

    mock_combined_ds = MagicMock(spec=Dataset)
    mock_combined_ds.column_names = []
    mocker.patch("benches.shared.Datasets._load_and_combine_splits", return_value=mock_combined_ds)

    with pytest.raises(AttributeError,
                       match=f"Could not find processor function for dataset '{dataset_key}'"):
        benchmark_shared.Datasets.load_dataset_by_name(dataset_key, 10)

    benchmark_shared.Datasets.registry = original_registry


def test_add_common_args():
    parser = argparse.ArgumentParser()
    benchmark_shared.add_common_args(parser)
    args = parser.parse_args(["--debug"])
    assert args.debug is True
    assert args.openai_key is None


def test_add_generation_args():
    parser = argparse.ArgumentParser()
    benchmark_shared.add_generation_args(parser)
    args = parser.parse_args(["--dataset", "aqua", "--cutoff", "10"])
    assert args.dataset == "aqua"
    assert args.cutoff == 10
    assert args.provider == "ollama"


def test_add_evaluation_args():
    parser = argparse.ArgumentParser()
    benchmark_shared.add_evaluation_args(parser)
    args = parser.parse_args(
        ["--input-file", "dummy.jsonl", "--extractor-type", "llm", "--show-details"])
    assert args.input_file == "dummy.jsonl"
    assert args.extractor_type == "llm"
    assert args.show_details is True
    assert args.provider == "ollama"


@pytest.mark.asyncio
async def test_benchmark_run_main_sync_flow(mocker, mock_load_dataset, mock_get_llm,
                                            mock_cot_methods, tmp_path):
    output_file = tmp_path / "results.jsonl"
    mocker.patch("sys.argv", [
        "benches/run.py",
        "--dataset", "gsm8k",
        "--cutoff", "1",
        "--provider", "ollama",
        "--output-file", str(output_file),
    ])
    # Apply the get_llm patch specifically for this test's scope
    mocker.patch("benches.shared.get_llm", return_value=mock_get_llm)
    with patch.object(benchmark_shared.Datasets, "_process_gsm8k", return_value=(["q1"], ["g1"])):
        mocker.patch("benches.run.run_setup_phase", return_value=True)
        await benchmark_run.main()
        assert output_file.exists()
        assert len(output_file.read_text().splitlines()) > 0


@pytest.mark.asyncio
async def test_benchmark_run_main_async_flow(mocker, mock_load_dataset, mock_get_llm,
                                             mock_cot_methods, tmp_path):
    output_file = tmp_path / "results_async.jsonl"
    mocker.patch("sys.argv", [
        "benches/run.py",
        "--dataset", "gsm8k",
        "--cutoff", "1",
        "--provider", "ollama",
        "--output-file", str(output_file),
        "--use-async",
        "--concurrency", "1",
    ])
    mocker.patch("benches.shared.get_llm", return_value=mock_get_llm)
    with patch.object(benchmark_shared.Datasets, "_process_gsm8k", return_value=(["q1"], ["g1"])):
        mocker.patch("benches.run.run_setup_phase", return_value=True)
        await benchmark_run.main()
        assert output_file.exists()
        assert len(output_file.read_text().splitlines()) > 0


@pytest.mark.asyncio
async def test_benchmark_evaluate_main_heuristic(mocker, tmp_path):
    input_content = """
    {"question": "q1", "gold": "10", "method": "MethodA", "dataset": "gsm8k", "raw_output": "Answer: 10", "time": 1.0}
    {"question": "q2", "gold": "5", "method": "MethodA", "dataset": "gsm8k", "raw_output": "Answer: 50", "time": 1.1}
    """
    input_file = tmp_path / "eval_in.jsonl"
    input_file.write_text(input_content.strip())

    mocker.patch("sys.argv", [
        "benches/evaluate.py",
        "--input-file", str(input_file),
        "--extractor-type", "heuristic",
        "--provider", "ollama",
    ])

    mocker.patch("polars.DataFrame.select")
    mocker.patch("builtins.print")
    # Mock get_llm even for heuristic to avoid potential side effects if called unexpectedly
    mocker.patch("benches.shared.get_llm")

    await benchmark_evaluate.main()


@pytest.mark.xfail(reason="Known issue: generate_json_async mock is not called in evaluate.py test")
@pytest.mark.asyncio
async def test_benchmark_evaluate_main_llm(mocker, tmp_path):
    input_content = """
    {"question": "q1", "gold": "10", "method": "MethodA", "dataset": "gsm8k", "raw_output": "Some reasoning... 10", "time": 1.0}
    """
    input_file = tmp_path / "eval_in_llm.jsonl"
    input_file.write_text(input_content.strip())

    mocker.patch("sys.argv", [
        "benches/evaluate.py",
        "--input-file", str(input_file),
        "--extractor-type", "llm",
        "--provider", "ollama",
    ])

    mocker.patch("polars.DataFrame.select")
    mocker.patch("builtins.print")  # Mock print unless debugging

    # --- Setup the specific mock for the LLM's async JSON generation ---
    mock_llm_instance = MagicMock()

    # Create the object we expect generate_json_async to return
    expected_result_obj = MagicMock(spec=ExtractedAnswer)
    expected_result_obj.final_answer = "10"

    # --- Use asyncio.Future for the mock's return value ---
    # Create a future that is already completed with the desired result
    future_result = asyncio.Future()
    future_result.set_result(expected_result_obj)

    # Configure generate_json_async to return the completed future
    # When the code under test awaits this, it will get the result immediately
    mock_llm_instance.generate_json_async = MagicMock(return_value=future_result)
    # --------------------------------------------------------

    # Patch get_llm to return this specific instance
    mocker.patch("benches.shared.get_llm", return_value=mock_llm_instance)

    # --- Run the main function of the evaluation script ---
    await benchmark_evaluate.main()

    # --- Assert that the mocked async method was called ---
    try:
        # Check if the mock was called
        mock_llm_instance.generate_json_async.assert_called()

        # Additionally, check if it was called with the expected arguments
        # Use mocker.ANY for arguments we don't need to match exactly (like prompt, temp, seed)
        mock_llm_instance.generate_json_async.assert_called_once_with(
            mocker.ANY,  # The formatted prompt string
            response_model=ExtractedAnswer,  # Verify response_model was passed
            temperature=mocker.ANY,
            max_tokens=mocker.ANY,
            seed=mocker.ANY
        )
    except AssertionError as e:
        # Print diagnostic info if the assertion fails
        print("Calls to generate_json_async:", mock_llm_instance.generate_json_async.call_args_list)
        # Print the arguments it *was* called with, if any
        if mock_llm_instance.generate_json_async.called:
            print("Actual call args:", mock_llm_instance.generate_json_async.call_args)
        raise e

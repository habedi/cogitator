import argparse
import asyncio
import logging
from unittest.mock import patch, MagicMock

import pytest
from datasets import Dataset

import benches.evaluate as benchmark_evaluate
import benches.run as benchmark_run
import benches.shared as benchmark_shared
from cogitator import ExtractedAnswer


@pytest.fixture(autouse=True)
def patch_sentence_transformer_init(mocker):
    mocker.patch("sentence_transformers.SentenceTransformer", return_value=MagicMock())


@pytest.fixture
def mock_load_dataset(mocker):
    mock_ds = MagicMock(spec=Dataset)
    mock_ds.__len__.return_value = 10
    mock_ds.column_names = ["question", "answer"]
    mock_ds.__getitem__.return_value = {"question": "q1", "answer": "a1"}
    mocker.patch("benches.shared.concatenate_datasets", return_value=mock_ds)
    return mocker.patch("benches.shared.load_dataset", return_value=mock_ds)


@pytest.fixture
def mock_llm(mocker):
    mock_llm_instance = MagicMock()
    mock_llm_instance.generate.return_value = "Mock LLM Response"
    mock_llm_instance.generate_json.return_value = MagicMock(final_answer="Mock JSON")

    async def async_gen_side_effect(*args, **kwargs):
        await asyncio.sleep(0)
        return "Mock Async LLM Response"

    mock_llm_instance.generate_async = MagicMock(side_effect=async_gen_side_effect)

    async def async_json_side_effect(*args, **kwargs):
        await asyncio.sleep(0)
        mock_answer = MagicMock(spec=ExtractedAnswer)
        mock_answer.final_answer = "Mock Async JSON"
        return mock_answer

    mock_llm_instance.generate_json_async = MagicMock(side_effect=async_json_side_effect)
    mock_llm_instance.get_last_prompt_tokens.return_value = 10
    mock_llm_instance.get_last_completion_tokens.return_value = 20
    return mock_llm_instance


@pytest.fixture
def mock_cot_methods(mocker):
    async def completed_awaitable_none(*args, **kwargs):
        await asyncio.sleep(0)
        return None

    async def auto_run_async_side_effect(*args, **kwargs):
        await asyncio.sleep(0)
        return "AutoCoT Mock Async Output"

    async def cdw_run_async_side_effect(*args, **kwargs):
        await asyncio.sleep(0)
        return "CDWCoT Mock Async Output"

    async def sc_run_async_side_effect(*args, **kwargs):
        await asyncio.sleep(0)
        return "SC Mock Async Output"

    async def ltm_run_async_side_effect(*args, **kwargs):
        await asyncio.sleep(0)
        return "LtM Mock Async Output"

    async def tot_run_async_side_effect(*args, **kwargs):
        await asyncio.sleep(0)
        return "ToT Mock Async Output"

    async def got_run_async_side_effect(*args, **kwargs):
        await asyncio.sleep(0)
        return "GoT Mock Async Output"

    mocker.patch("cogitator.AutoCoT.fit")
    mocker.patch("cogitator.AutoCoT.fit_async", side_effect=completed_awaitable_none)
    mocker.patch("cogitator.AutoCoT.run", return_value="AutoCoT Mock Output")
    mocker.patch("cogitator.AutoCoT.run_async", side_effect=auto_run_async_side_effect)
    mocker.patch("cogitator.CDWCoT.init_pool")
    mocker.patch("cogitator.CDWCoT.init_pool_async", side_effect=completed_awaitable_none)
    mocker.patch("cogitator.CDWCoT.train")
    mocker.patch("cogitator.CDWCoT.train_async", side_effect=completed_awaitable_none)
    mocker.patch("cogitator.CDWCoT.run", return_value="CDWCoT Mock Output")
    mocker.patch("cogitator.CDWCoT.run_async", side_effect=cdw_run_async_side_effect)
    mocker.patch("cogitator.SelfConsistency.run", return_value="SC Mock Output")
    mocker.patch("cogitator.SelfConsistency.run_async", side_effect=sc_run_async_side_effect)
    mocker.patch("cogitator.LeastToMost.run", return_value="LtM Mock Output")
    mocker.patch("cogitator.LeastToMost.run_async", side_effect=ltm_run_async_side_effect)
    mocker.patch("cogitator.TreeOfThoughts.run", return_value="ToT Mock Output")
    mocker.patch("cogitator.TreeOfThoughts.run_async", side_effect=tot_run_async_side_effect)
    mocker.patch("cogitator.GraphOfThoughts.run", return_value="GoT Mock Output")
    mocker.patch("cogitator.GraphOfThoughts.run_async", side_effect=got_run_async_side_effect)


@pytest.fixture
def mock_st_embeddings(mocker):
    mock_st_instance = MagicMock()
    mock_st_instance.encode.return_value = [MagicMock()]
    mocker.patch("sentence_transformers.SentenceTransformer", return_value=mock_st_instance)


DEFAULT_PROVIDER = benchmark_shared.DEFAULT_PROVIDER
DEFAULT_DATASET = benchmark_shared.DEFAULT_DATASET
DEFAULT_CUTOFF = benchmark_shared.DEFAULT_CUTOFF
DEFAULT_CONCURRENCY = benchmark_shared.DEFAULT_CONCURRENCY
DEFAULT_OUTPUT_FILE = benchmark_shared.DEFAULT_OUTPUT_FILE
DEFAULT_EXTRACTOR_TYPE = benchmark_shared.DEFAULT_EXTRACTOR_TYPE


@pytest.fixture
def mock_args_namespace_factory():
    def _create(**kwargs):
        defaults = {'openai_key': None, 'debug': False, 'dataset': DEFAULT_DATASET,
                    'cutoff': DEFAULT_CUTOFF, 'provider': DEFAULT_PROVIDER, 'model_name': None,
                    'use_async': False, 'concurrency': DEFAULT_CONCURRENCY,
                    'use_json_strategies': False, 'output_file': DEFAULT_OUTPUT_FILE,
                    'input_file': None, 'extractor_type': DEFAULT_EXTRACTOR_TYPE,
                    'show_details': False}
        defaults.update(kwargs)
        return argparse.Namespace(**defaults)

    return _create


@pytest.fixture
def generation_parser():
    parser = argparse.ArgumentParser()
    benchmark_shared.add_common_args(parser)
    benchmark_shared.add_generation_args(parser)
    return parser


@pytest.fixture
def evaluation_parser():
    parser = argparse.ArgumentParser()
    benchmark_shared.add_common_args(parser)
    benchmark_shared.add_evaluation_args(parser)
    return parser


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
async def test_benchmark_run_main_sync_flow(mocker, mock_load_dataset, mock_llm,
                                            mock_cot_methods, mock_st_embeddings, tmp_path):
    output_file = tmp_path / "results.jsonl"
    config_file = tmp_path / "bench_config.yml"
    config_file.touch()
    mocker.patch("benches.shared.DEFAULT_BENCH_CONFIG_PATH", str(config_file))
    mocker.patch("sys.argv",
                 ["benches/run.py", "--dataset", "gsm8k", "--cutoff", "1", "--provider", "ollama",
                  "--output-file", str(output_file)])
    mocker.patch("benches.run.get_llm", return_value=mock_llm)
    with patch.object(benchmark_shared.Datasets, "_process_gsm8k", return_value=(["q1"], ["g1"])):
        mocker.patch("benches.run.run_setup_phase", return_value=True)
        await benchmark_run.main()
        assert output_file.exists()
        assert len(output_file.read_text().splitlines()) > 0


@pytest.mark.asyncio
async def test_benchmark_run_main_async_flow(mocker, mock_load_dataset, mock_llm,
                                             mock_cot_methods, mock_st_embeddings, tmp_path):
    output_file = tmp_path / "results_async.jsonl"
    config_file = tmp_path / "bench_config_async.yml"
    config_file.touch()
    mocker.patch("benches.shared.DEFAULT_BENCH_CONFIG_PATH", str(config_file))
    mocker.patch("sys.argv",
                 ["benches/run.py", "--dataset", "gsm8k", "--cutoff", "1", "--provider", "ollama",
                  "--output-file", str(output_file), "--use-async", "--concurrency", "1"])
    mocker.patch("benches.run.get_llm", return_value=mock_llm)
    with patch.object(benchmark_shared.Datasets, "_process_gsm8k", return_value=(["q1"], ["g1"])):
        mocker.patch("benches.run.run_setup_phase", return_value=True)
        await benchmark_run.main()
        assert output_file.exists()
        assert len(output_file.read_text().splitlines()) > 0


@pytest.mark.asyncio
async def test_benchmark_evaluate_main_heuristic(mocker, tmp_path):
    input_content = """{"question": "q1", "gold": "10", "method": "MethodA", "dataset": "gsm8k", "raw_output": "Answer: 10", "time": 1.0}"""
    input_file = tmp_path / "eval_in.jsonl"
    input_file.write_text(input_content.strip())
    config_file = tmp_path / "bench_config_eval_h.yml"
    config_file.touch()
    mocker.patch("benches.shared.DEFAULT_BENCH_CONFIG_PATH", str(config_file))
    mocker.patch("sys.argv",
                 ["benches/evaluate.py", "--input-file", str(input_file), "--extractor-type",
                  "heuristic", "--provider", "ollama"])
    mocker.patch("polars.DataFrame.select")
    mocker.patch("builtins.print")
    mocker.patch("benches.evaluate.get_llm")
    await benchmark_evaluate.main()


@pytest.mark.asyncio
async def test_benchmark_evaluate_main_llm(mocker, mock_llm, tmp_path):
    input_content = """{"question": "q1", "gold": "10", "method": "MethodA", "dataset": "gsm8k", "raw_output": "Some reasoning... 10", "time": 1.0}"""
    input_file = tmp_path / "eval_in_llm.jsonl"
    input_file.write_text(input_content.strip())
    config_file = tmp_path / "bench_config_eval_llm.yml"
    config_file.touch()
    mocker.patch("benches.shared.DEFAULT_BENCH_CONFIG_PATH", str(config_file))
    mocker.patch("sys.argv",
                 ["benches/evaluate.py", "--input-file", str(input_file), "--extractor-type", "llm",
                  "--provider", "ollama"])
    mocker.patch("polars.DataFrame.select")
    mocker.patch("builtins.print")

    mocker.patch("benches.evaluate.get_llm", return_value=mock_llm)

    await benchmark_evaluate.main()

    mock_llm.generate_json_async.assert_called()

    # More specific check if needed, using mocker.ANY for potentially variable args
    mock_llm.generate_json_async.assert_called_once_with(
        mocker.ANY,  # The prompt string
        response_model=ExtractedAnswer,
        temperature=mocker.ANY,
        max_tokens=mocker.ANY,
        seed=mocker.ANY
    )


def test_load_config_no_file_defaults(mock_args_namespace_factory, generation_parser):
    args = mock_args_namespace_factory()
    config_path = "non_existent_file.yml"
    merged_config = benchmark_shared.load_and_merge_config(args, generation_parser, config_path,
                                                           "generation")
    assert merged_config['provider'] == DEFAULT_PROVIDER
    assert merged_config['dataset'] == DEFAULT_DATASET
    assert merged_config['use_async'] is False
    assert 'strategies' in merged_config and not merged_config['strategies']


def test_load_config_empty_file_defaults(tmp_path, mock_args_namespace_factory, generation_parser):
    args = mock_args_namespace_factory()
    config_file = tmp_path / "empty.yml"
    config_file.touch()
    merged_config = benchmark_shared.load_and_merge_config(args, generation_parser,
                                                           str(config_file), "generation")
    assert merged_config['provider'] == DEFAULT_PROVIDER
    assert merged_config['dataset'] == DEFAULT_DATASET


def test_load_config_from_yaml(tmp_path, mock_args_namespace_factory, generation_parser):
    args = mock_args_namespace_factory()
    yaml_content = """
common:
  debug: true
generation:
  provider: openai
  model_name: gpt-override
  use_async: true
  llm_params: {seed: 123}
  output_file: yaml_output.jsonl
strategies:
  AutoCoT: {n_demos: 3, enabled: true}
  CDWCoT: {enabled: false}
    """
    config_file = tmp_path / "test.yml"
    config_file.write_text(yaml_content)
    merged_config = benchmark_shared.load_and_merge_config(args, generation_parser,
                                                           str(config_file), "generation")
    assert merged_config['provider'] == "openai"
    assert merged_config['model_name'] == "gpt-override"
    assert merged_config['use_async'] is True
    assert merged_config['debug'] is True
    assert merged_config['llm_params']['seed'] == 123
    assert merged_config['output_file'] == "yaml_output.jsonl"
    assert merged_config['strategies']['AutoCoT']['n_demos'] == 3
    assert merged_config['strategies']['CDWCoT']['enabled'] is False
    assert merged_config['cutoff'] == DEFAULT_CUTOFF


def test_load_config_cli_overrides_yaml(tmp_path, mock_args_namespace_factory, generation_parser):
    args = mock_args_namespace_factory(provider="openai", use_async=False, dataset="csqa")
    yaml_content = """
generation:
  provider: some_yaml_provider
  use_async: false
  cutoff: 99
  dataset: aqua
    """
    config_file = tmp_path / "test_override.yml"
    config_file.write_text(yaml_content)
    merged_config = benchmark_shared.load_and_merge_config(args, generation_parser,
                                                           str(config_file), "generation")
    assert merged_config['provider'] == "openai"
    assert merged_config['use_async'] is False
    assert merged_config['cutoff'] == 99
    assert merged_config['dataset'] == "csqa"


def test_load_config_evaluation_section(tmp_path, mock_args_namespace_factory, evaluation_parser):
    args = mock_args_namespace_factory()
    yaml_content = """
generation: {output_file: gen_out.jsonl}
evaluation:
  input_file: eval_in.jsonl
  extractor:
    type: llm
    provider: openai
    model_name: gpt-4-eval
  show_details: true
    """
    config_file = tmp_path / "test_eval.yml"
    config_file.write_text(yaml_content)
    merged_config = benchmark_shared.load_and_merge_config(args, evaluation_parser,
                                                           str(config_file), "evaluation")
    assert merged_config['input_file'] == "eval_in.jsonl"
    assert merged_config['extractor_type'] == "llm"
    assert merged_config['extractor_provider'] == "openai"
    assert merged_config['extractor_model_name'] == "gpt-4-eval"
    assert merged_config['show_details'] is True


def test_load_config_evaluation_cli_override(tmp_path, mock_args_namespace_factory,
                                             evaluation_parser):
    args = mock_args_namespace_factory(provider="openai", extractor_type="llm")
    yaml_content = """
evaluation:
  extractor:
    type: heuristic
    provider: ollama
    model_name: yaml-model
    """
    config_file = tmp_path / "test_eval_cli.yml"
    config_file.write_text(yaml_content)
    merged_config = benchmark_shared.load_and_merge_config(args, evaluation_parser,
                                                           str(config_file), "evaluation")
    assert merged_config['extractor_type'] == "llm"
    assert merged_config['extractor_provider'] == "openai"
    assert merged_config['extractor_model_name'] == "yaml-model"


def test_load_config_evaluation_default_input(tmp_path, mock_args_namespace_factory,
                                              evaluation_parser):
    args = mock_args_namespace_factory()
    yaml_content = """
generation: {output_file: gen_out_for_eval.jsonl}
evaluation: {extractor: {type: heuristic}}
    """
    config_file = tmp_path / "test_eval_default_in.yml"
    config_file.write_text(yaml_content)
    merged_config = benchmark_shared.load_and_merge_config(args, evaluation_parser,
                                                           str(config_file), "evaluation")
    assert merged_config['input_file'] == "gen_out_for_eval.jsonl"
    assert merged_config['extractor_type'] == "heuristic"


def test_load_config_malformed_yaml(tmp_path, mock_args_namespace_factory, generation_parser, caplog):
    args = mock_args_namespace_factory()
    yaml_content = "generation: { provider: openai, model_name: [invalid"
    config_file = tmp_path / "malformed.yml"
    config_file.write_text(yaml_content)
    with caplog.at_level(logging.WARNING):
        merged_config = benchmark_shared.load_and_merge_config(args, generation_parser,
                                                               str(config_file), "generation")
    assert "Error parsing" in caplog.text
    assert config_file.name in caplog.text
    assert "Using command-line arguments and defaults" in caplog.text
    assert merged_config['provider'] == DEFAULT_PROVIDER
    assert merged_config['dataset'] == DEFAULT_DATASET
    assert not merged_config['strategies']

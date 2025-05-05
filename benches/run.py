import argparse
import asyncio
import json
import logging
import os
import sys
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

from benches.shared import (
    setup_logging, Datasets, get_llm,
    add_common_args, add_generation_args, load_and_merge_config,
    DEFAULT_OPENAI_ENV_VAR
)
from cogitator import (
    AutoCoT, CDWCoT, GraphOfThoughts, LeastToMost, SelfConsistency,
    TreeOfThoughts, BaseLLM
)

logger = logging.getLogger("benchmark_run")


def _get_strategy_config(config: Dict[str, Any], name: str) -> Dict[str, Any]:
    return config.get("strategies", {}).get(name, {})


def _is_strategy_enabled(config: Dict[str, Any], name: str) -> bool:
    strategy_cfg = _get_strategy_config(config, name)
    return strategy_cfg is not None and strategy_cfg.get("enabled", True)


async def run_setup_phase(
    config: Dict[str, Any],
    llm: BaseLLM, methods_to_run: List[str],
    questions: List[str], golds: List[str], semaphore: asyncio.Semaphore,
    instances: Dict[str, Any]
) -> bool:
    logger.info("Running setup phase (fit/train) using config...")
    needs_auto_setup = "AutoCoT" in methods_to_run and "AutoCoT" in instances
    needs_cdw_setup = "CDWCoT" in methods_to_run and "CDWCoT" in instances

    if not needs_auto_setup and not needs_cdw_setup:
        logger.info("No methods requiring setup found or enabled. Skipping setup phase.")
        return True

    auto_instance = instances.get("AutoCoT")
    cdw_instance = instances.get("CDWCoT")

    try:
        setup_tasks = []
        if needs_auto_setup and auto_instance:
            if config['use_async']:
                logger.info("Scheduling AutoCoT fit_async...")
                setup_tasks.append(auto_instance.fit_async(questions, semaphore=semaphore))
            else:
                logger.info("Running AutoCoT fit...")
                auto_instance.fit(questions)
                logger.info("AutoCoT fit complete.")

        if needs_cdw_setup and cdw_instance:
            cdw_cfg = _get_strategy_config(config, "CDWCoT")
            train_cfg = cdw_cfg.get("train_params", {})
            train_epochs = train_cfg.get('epochs', 5)
            train_patience = train_cfg.get('patience', 2)
            train_val_split = train_cfg.get('val_split', 0.2)
            logger.info(
                f"CDWCoT Train Params: epochs={train_epochs}, patience={train_patience}, val_split={train_val_split}")

            if config['use_async']:
                async def cdw_setup_wrapper():
                    logger.info("Scheduling CDWCoT init_pool_async...")
                    await cdw_instance.init_pool_async(questions, golds, semaphore=semaphore)
                    logger.info("CDWCoT init_pool_async complete. Scheduling train_async...")
                    await cdw_instance.train_async(
                        epochs=train_epochs, patience=train_patience, val_split=train_val_split,
                        semaphore=semaphore
                    )
                    logger.info("CDWCoT train_async complete.")

                setup_tasks.append(cdw_setup_wrapper())
            else:
                logger.info("Running CDWCoT init_pool...")
                cdw_instance.init_pool(questions, golds)
                logger.info("CDWCoT init_pool complete. Running train...")
                cdw_instance.train(
                    epochs=train_epochs, patience=train_patience, val_split=train_val_split
                )
                logger.info("CDWCoT train complete.")

        if config['use_async'] and setup_tasks:
            logger.info(f"Awaiting {len(setup_tasks)} async setup tasks...")
            await asyncio.gather(*setup_tasks)
            logger.info("Async setup tasks complete.")
        logger.info("Setup phase complete.")
        return True
    except Exception as e:
        logger.error(f"Error during setup phase (fit/train): {e}", exc_info=True)
        logger.warning("Setup failed. Methods requiring setup may be skipped or fail.")
        return False


def get_methods_to_run(
    config: Dict[str, Any],
    instances: Dict[str, Any],
    llm: BaseLLM
) -> List[Tuple[str, Optional[Callable]]]:
    methods = []
    use_async = config.get('use_async', False)
    global_llm_params = config.get('llm_params', {})

    if _is_strategy_enabled(config, "ZeroShotCoT"):
        zero_shot_prompt = lambda q: f"Q: {q}\nA: Let's think step by step."
        zero_shot_kwargs = global_llm_params.copy()
        zero_shot_func = (
            lambda q, **kw: llm.generate_async(zero_shot_prompt(q), **zero_shot_kwargs,
                                               **kw)) if use_async else \
            (lambda q, **kw: llm.generate(zero_shot_prompt(q), **zero_shot_kwargs, **kw))
        methods.append(("Zero-Shot-CoT", zero_shot_func))
        logger.debug("ZeroShotCoT is enabled.")

    for name, instance in instances.items():
        if _is_strategy_enabled(config, name) and instance:
            strategy_cfg = _get_strategy_config(config, name)
            run_kwargs = global_llm_params.copy()
            wrapped_run_method: Optional[Callable] = None

            if name == "SelfConsistency":
                sc_temp = strategy_cfg.get('temperature')
                if sc_temp is not None: run_kwargs['temperature'] = sc_temp
                sc_stop = strategy_cfg.get('stop')
                if sc_stop is not None: run_kwargs['stop'] = sc_stop
                run_method_base = instance.run_async if use_async else instance.run
                wrapped_run_method = lambda q, rk=run_kwargs: run_method_base(q, **rk)

            elif name == "GraphOfThoughts":
                goo_list = strategy_cfg.get('graph_of_operations')
                if not goo_list or not isinstance(goo_list, list):
                    logger.error(
                        f"GraphOfThoughts strategy enabled but 'graph_of_operations'"
                        f" is missing or invalid in config. Skipping.")
                    wrapped_run_method = None
                else:
                    logger.info(f"GraphOfThoughts configured with GoO: {goo_list}")
                    if use_async:
                        run_method_base = instance.run_async
                        wrapped_run_method = lambda q, goo=goo_list, \
                                                    rk=run_kwargs: run_method_base(q,
                                                                                   graph_of_operations=goo,
                                                                                   **rk)
                    else:
                        logger.warning(
                            "GraphOfThoughts sync run is not supported, skipping sync execution.")
                        wrapped_run_method = None
            else:
                run_method_base = instance.run_async if use_async else instance.run
                wrapped_run_method = lambda q, rk=run_kwargs: run_method_base(q, **rk)

            if wrapped_run_method is not None:
                methods.append((name, wrapped_run_method))
                logger.debug(f"{name} is enabled.")
        else:
            logger.info(
                f"Skipping strategy '{name}' as it is disabled in config or instance is missing.")

    return methods


async def run_single_async(q: str, model_func: Callable, semaphore: asyncio.Semaphore) \
    -> Tuple[str, float]:
    t0 = time.time()
    raw_output = "[ERROR]"
    try:
        async with semaphore:
            raw_output = await model_func(q)
    except Exception as e:
        logger.error(f"Error during async generation: {e}", exc_info=True)
        raw_output = f"[ERROR: {type(e).__name__}]"
    time_taken = time.time() - t0
    return str(raw_output) if raw_output is not None else "[ERROR]", time_taken


def run_single_sync(q: str, model_func: Callable) -> Tuple[str, float]:
    t0 = time.time()
    raw_output = "[ERROR]"
    try:
        raw_output = model_func(q)
    except Exception as e:
        logger.error(f"Error during sync generation: {e}", exc_info=True)
        raw_output = f"[ERROR: {type(e).__name__}]"
    time_taken = time.time() - t0
    return str(raw_output) if raw_output is not None else "[ERROR]", time_taken


async def main():
    parser = argparse.ArgumentParser(description="Run Cogitator Benchmarks - Generation Phase")
    add_common_args(parser)
    add_generation_args(parser)
    args = parser.parse_args()

    config = load_and_merge_config(args, parser, config_section="generation")

    log_level = logging.DEBUG if config['debug'] else logging.INFO
    setup_logging(log_level)
    logger.info(f"Running Generation Phase with effective config: {config}")

    try:
        questions, golds = Datasets.load_dataset_by_name(config['dataset'], config['cutoff'])
    except Exception as e:
        logger.error(f"Failed to load dataset '{config['dataset']}': {e}", exc_info=True)
        sys.exit(1)

    try:
        key_from_cli = getattr(args, 'openai_key', None)
        env_var_name = config.get('openai_key_env_var', DEFAULT_OPENAI_ENV_VAR)
        key_from_env = os.getenv(env_var_name) if env_var_name else None
        openai_api_key = key_from_cli or key_from_env

        llm = get_llm(
            provider=config['provider'],
            model_name=config['model_name'],
            openai_key=openai_api_key,
            ollama_host=config.get('ollama_host', None),
            llm_params=config.get('llm_params', {})
        )
    except Exception as e:
        logger.error(f"Failed to initialize LLM: {e}", exc_info=True)
        sys.exit(1)

    logger.info("Initializing CoT strategies based on config...")
    instances = {}
    global_llm_params = config.get('llm_params', {})
    global_seed = global_llm_params.get('seed')
    global_max_tokens = global_llm_params.get('max_tokens')

    def extract_init_params(cfg: Dict[str, Any], keys: List[str]) -> Dict[str, Any]:
        return {k: cfg[k] for k in keys if k in cfg}

    if _is_strategy_enabled(config, "AutoCoT"):
        cfg = _get_strategy_config(config, "AutoCoT")
        params = extract_init_params(cfg,
                                     ["n_demos", "max_q_tokens", "max_steps", "prompt_template",
                                      "max_retries"])
        instances["AutoCoT"] = AutoCoT(llm,
                                       max_tokens=global_max_tokens,
                                       rand_seed=global_seed,
                                       **params)
        logger.info(f"Initialized AutoCoT with params: {params}")

    if _is_strategy_enabled(config, "CDWCoT"):
        cfg = _get_strategy_config(config, "CDWCoT")
        params = extract_init_params(cfg, ["pool_size", "n_clusters", "lr", "sample_size",
                                           "max_grad_norm", "init_pool_retries"])
        instances["CDWCoT"] = CDWCoT(llm,
                                     max_tokens=global_max_tokens,
                                     seed=global_seed,
                                     **params)
        logger.info(f"Initialized CDWCoT with params: {params}")

    if _is_strategy_enabled(config, "SelfConsistency"):
        cfg = _get_strategy_config(config, "SelfConsistency")
        params = extract_init_params(cfg, ["n_samples"])
        sc_extraction_format = cfg.get("internal_extraction_format")
        if sc_extraction_format is None:
            sc_extraction_format = "json" if config['use_json_strategies'] else "heuristic"
        params['internal_extraction_format'] = sc_extraction_format
        params['seed'] = global_seed
        params['max_tokens'] = global_max_tokens
        instances["SelfConsistency"] = SelfConsistency(llm, **params)
        logger.info(f"Initialized SelfConsistency with params: {params}")

    if _is_strategy_enabled(config, "LeastToMost"):
        cfg = _get_strategy_config(config, "LeastToMost")
        params = extract_init_params(cfg, ["max_subqs", "decompose_prompt_template",
                                           "solve_prompt_template", "final_answer_prompt_template"])
        ltm_format = cfg.get("intermediate_output_format")
        if ltm_format is None:
            ltm_format = "json" if config['use_json_strategies'] else "text"
        params['intermediate_output_format'] = ltm_format
        instances["LeastToMost"] = LeastToMost(llm, max_tokens=global_max_tokens, seed=global_seed,
                                               **params)
        logger.info(f"Initialized LeastToMost with params: {params}")

    if _is_strategy_enabled(config, "TreeOfThoughts"):
        cfg = _get_strategy_config(config, "TreeOfThoughts")
        params = extract_init_params(cfg, ["max_depth", "num_branches", "sims", "c_puct",
                                           "expand_prompt", "eval_prompt"])
        instances["TreeOfThoughts"] = TreeOfThoughts(llm, max_tokens=global_max_tokens,
                                                     seed=global_seed, **params)
        logger.info(f"Initialized TreeOfThoughts with params: {params}")

    if _is_strategy_enabled(config, "GraphOfThoughts"):
        cfg = _get_strategy_config(config, "GraphOfThoughts")
        params = extract_init_params(cfg, ["final_answer_format", "prompts"])

        got_format = cfg.get("final_answer_format")

        if got_format is None:
            got_format = "json" if config.get('use_json_strategies', False) else "text"
        params['final_answer_format'] = got_format

        if "prompts" not in params and "prompts" in cfg:
            params["prompts"] = cfg["prompts"]

        instances["GraphOfThoughts"] = GraphOfThoughts(
            llm,
            max_tokens=global_max_tokens,
            seed=global_seed,
            **params
        )
        logger.info(f"Initialized GraphOfThoughts with params: {params}")

    all_methods_to_run = get_methods_to_run(config, instances, llm)
    methods_to_run_names = [name for name, func in all_methods_to_run if func is not None]

    if not all_methods_to_run:
        logger.error("No strategies are enabled or initialized. Exiting.")
        sys.exit(1)

    semaphore = asyncio.Semaphore(config['concurrency'])
    setup_successful = await run_setup_phase(config, llm, methods_to_run_names, questions, golds,
                                             semaphore, instances)

    output_filepath = config['output_file']
    logger.info(
        f"Starting generation for {len(methods_to_run_names)} enabled methods, saving to {output_filepath}...")

    with open(output_filepath, 'w') as outfile:
        if config['use_async']:
            tasks = []
            task_info = []
            for i, q in enumerate(questions):
                for name, model_func_wrapped in all_methods_to_run:
                    if model_func_wrapped is None: continue
                    tasks.append(run_single_async(q, model_func_wrapped, semaphore))
                    task_info.append((i, q, name))

            logger.info(f"Running {len(tasks)} async generation tasks...")
            gen_outputs = await asyncio.gather(*tasks)
            logger.info("Async generation tasks complete.")

            for task_idx, (raw_output, time_taken) in enumerate(gen_outputs):
                i, q, name = task_info[task_idx]
                record = {"question": q, "gold": golds[i], "method": name,
                          "dataset": config['dataset'], "raw_output": raw_output,
                          "time": time_taken}
                outfile.write(json.dumps(record) + '\n')
                logger.debug(f"Generated: Q{i + 1} - {name} ({time_taken:.2f}s)")

        else:
            for i, q in enumerate(questions):
                logger.info(f"Processing Question {i + 1}/{len(questions)}")
                for name, model_func_wrapped in all_methods_to_run:
                    if model_func_wrapped is None: continue

                    if name == "AutoCoT" and (
                        not setup_successful or not getattr(instances.get("AutoCoT"), 'demos',
                                                            None)):
                        logger.warning(f"Skipping {name} for Q{i + 1} (setup failed/incomplete).")
                        continue
                    if name == "CDWCoT" and (
                        not setup_successful or not getattr(instances.get("CDWCoT"), 'PC', None)):
                        logger.warning(f"Skipping {name} for Q{i + 1} (setup failed/incomplete).")
                        continue

                    logger.info(f"Running: Q{i + 1} - {name}")
                    raw_output, time_taken = run_single_sync(q, model_func_wrapped)
                    record = {"question": q, "gold": golds[i], "method": name,
                              "dataset": config['dataset'],
                              "raw_output": raw_output, "time": time_taken}
                    outfile.write(json.dumps(record) + '\n')

    logger.info(f"Generation complete. Raw results saved to {output_filepath}")


if __name__ == "__main__":
    asyncio.run(main())

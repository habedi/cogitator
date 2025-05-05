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
    """Gets the configuration for a specific strategy."""
    return config.get("strategies", {}).get(name, {})


def _is_strategy_enabled(config: Dict[str, Any], name: str) -> bool:
    """Checks if a strategy is enabled in the configuration."""
    strategy_cfg = _get_strategy_config(config, name)
    # Enabled by default if the section exists, unless 'enabled: false' is explicitly set
    return strategy_cfg is not None and strategy_cfg.get("enabled", True)


async def run_setup_phase(
    config: Dict[str, Any],
    llm: BaseLLM, methods_to_run: List[str],
    questions: List[str], golds: List[str], semaphore: asyncio.Semaphore,
    instances: Dict[str, Any]
) -> bool:
    """Runs the one-time setup (fit/train) for strategies that require it."""
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
                # Pass semaphore to fit_async if it accepts it (check AutoCoT implementation)
                # Assuming fit_async accepts semaphore:
                setup_tasks.append(auto_instance.fit_async(questions, semaphore=semaphore))
            else:
                logger.info("Running AutoCoT fit...")
                auto_instance.fit(questions)
                logger.info("AutoCoT fit complete.")

        if needs_cdw_setup and cdw_instance:
            cdw_cfg = _get_strategy_config(config, "CDWCoT")
            train_cfg = cdw_cfg.get("train_params", {})
            train_epochs = train_cfg.get('epochs', 5)  # Default epochs if not specified
            train_patience = train_cfg.get('patience', 2)  # Default patience
            train_val_split = train_cfg.get('val_split', 0.2)  # Default validation split
            logger.info(
                f"CDWCoT Train Params: epochs={train_epochs}, patience={train_patience}, val_split={train_val_split}")

            if config['use_async']:
                async def cdw_setup_wrapper():
                    logger.info("Scheduling CDWCoT init_pool_async...")
                    await cdw_instance.init_pool_async(questions, golds, semaphore=semaphore)
                    logger.info("CDWCoT init_pool_async complete. Scheduling train_async...")
                    await cdw_instance.train_async(
                        epochs=train_epochs, patience=train_patience, val_split=train_val_split,
                        semaphore=semaphore  # Pass semaphore here too
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

        # Wait for async setup tasks if any were scheduled
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
    """Creates runnable functions (lambdas) for each enabled strategy."""
    methods = []
    use_async = config.get('use_async', False)
    global_llm_params = config.get('llm_params', {})

    # --- Zero-Shot-CoT Baseline ---
    if _is_strategy_enabled(config, "ZeroShotCoT"):
        zero_shot_prompt = lambda q: f"Q: {q}\nA: Let's think step by step."
        # Combine global params with potential strategy-specific overrides if needed
        zero_shot_kwargs = global_llm_params.copy()
        # Example override (if you wanted ZeroShotCoT to use different temp):
        # zs_cfg = _get_strategy_config(config, "ZeroShotCoT")
        # zs_temp = zs_cfg.get('temperature')
        # if zs_temp is not None: zero_shot_kwargs['temperature'] = zs_temp

        # Create the callable function
        zero_shot_func = (
            lambda q, **kw: llm.generate_async(zero_shot_prompt(q), **zero_shot_kwargs, **kw)
        ) if use_async else (
            lambda q, **kw: llm.generate(zero_shot_prompt(q), **zero_shot_kwargs, **kw)
        )
        methods.append(("Zero-Shot-CoT", zero_shot_func))
        logger.debug("ZeroShotCoT is enabled.")

    # --- Other Strategies ---
    for name, instance in instances.items():
        if _is_strategy_enabled(config, name) and instance:
            strategy_cfg = _get_strategy_config(config, name)
            # Start with global LLM params, then override with strategy-specific ones
            run_kwargs = global_llm_params.copy()
            strategy_llm_params = strategy_cfg.get('llm_params', {})
            run_kwargs.update(strategy_llm_params)  # Apply strategy overrides

            wrapped_run_method: Optional[Callable] = None

            # --- Strategy-Specific Argument Handling ---

            if name == "SelfConsistency":
                # SC takes temperature, stop directly in run() args, handled via run_kwargs
                sc_temp = strategy_cfg.get('temperature')  # Check specific config first
                if sc_temp is not None: run_kwargs['temperature'] = sc_temp
                sc_stop = strategy_cfg.get('stop')
                if sc_stop is not None: run_kwargs['stop'] = sc_stop

                # Base method selection (sync/async)
                run_method_base = instance.run_async if use_async else instance.run
                # Lambda captures current run_kwargs (rk)
                current_kwargs = run_kwargs.copy()
                wrapped_run_method = lambda q_arg: run_method_base(
                    q_arg, **current_kwargs
                )


            elif name == "GraphOfThoughts":
                goo_list = strategy_cfg.get('graph_of_operations')
                if not goo_list or not isinstance(goo_list, list):
                    logger.error(
                        f"GraphOfThoughts strategy enabled but 'graph_of_operations'"
                        f" is missing or invalid in config. Skipping.")
                    wrapped_run_method = None
                else:
                    logger.info(f"GraphOfThoughts configured with GoO: {goo_list}")
                    # Capture necessary variables from the current scope
                    current_goo = goo_list
                    current_instance = instance
                    current_kwargs = run_kwargs.copy()  # Use a copy

                    if use_async:
                        # Define lambda that calls run_async with captured goo and kwargs
                        wrapped_run_method = lambda q_arg: current_instance.run_async(
                            q_arg, current_goo, **current_kwargs
                        )
                    else:
                        # Define lambda that calls run with captured goo and kwargs
                        wrapped_run_method = lambda q_arg: current_instance.run(
                            q_arg, current_goo, **current_kwargs
                        )
                    # Note: The sync case will correctly hit the NotImplementedError inside GoT's run method.


            else:  # Default handling for other strategies (AutoCoT, CDWCoT, LtM, ToT)
                # Assumes their run/run_async methods take (question, **kwargs)
                run_method_base = instance.run_async if use_async else instance.run
                current_kwargs = run_kwargs.copy()
                wrapped_run_method = lambda q_arg: run_method_base(
                    q_arg, **current_kwargs
                )

            # --- Add the wrapped method to the list ---
            if wrapped_run_method is not None:
                methods.append((name, wrapped_run_method))
                logger.debug(f"{name} is enabled.")
        else:
            logger.info(
                f"Skipping strategy '{name}' as it is disabled in config or instance is missing.")

    return methods


async def run_single_async(q: str, model_func: Callable, llm: BaseLLM, semaphore: asyncio.Semaphore) \
    -> Tuple[str, float, Optional[int], Optional[int]]:  # Modified return type
    """Runs a single strategy asynchronously and returns output, time, and token counts."""
    t0 = time.time()
    raw_output = "[ERROR]"
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    try:
        async with semaphore:
            # model_func is the lambda created in get_methods_to_run
            raw_output = await model_func(q)
        # Get token counts AFTER the call completes using the shared llm instance
        prompt_tokens = llm.get_last_prompt_tokens()
        completion_tokens = llm.get_last_completion_tokens()
    except Exception as e:
        # Log error but capture result as error string for JSONL
        logger.error(f"Error during async generation: {e}", exc_info=True)
        raw_output = f"[ERROR: {type(e).__name__}]"
        # Reset tokens on error as they might be from a previous successful run
        llm._reset_token_counts()
        prompt_tokens = None
        completion_tokens = None
    time_taken = time.time() - t0
    # Return counts along with output and time
    return str(
        raw_output) if raw_output is not None else "[ERROR]", time_taken, prompt_tokens, completion_tokens


def run_single_sync(q: str, model_func: Callable, llm: BaseLLM, strategy_name: str) -> Tuple[
    str, float, Optional[int], Optional[int]]:  # Modified return type
    """Runs a single strategy synchronously and returns output, time, and token counts."""
    t0 = time.time()
    raw_output = "[ERROR]"
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    try:
        # model_func is the lambda created in get_methods_to_run
        raw_output = model_func(q)
        # Get token counts AFTER the call completes
        prompt_tokens = llm.get_last_prompt_tokens()
        completion_tokens = llm.get_last_completion_tokens()
    except NotImplementedError as nie:
        # Handle expected error for GoT sync
        if strategy_name == "GraphOfThoughts":
            logger.warning(f"Caught expected NotImplementedError for {strategy_name}: {nie}")
            raw_output = f"[ERROR: {type(nie).__name__} - Sync not supported]"
        else:  # Log unexpected NotImplementedError for other strategies
            logger.error(f"Caught unexpected NotImplementedError for {strategy_name}: {nie}",
                         exc_info=True)
            raw_output = f"[ERROR: {type(nie).__name__}]"
        # Reset tokens on error
        llm._reset_token_counts()
        prompt_tokens = None
        completion_tokens = None
    except Exception as e:
        logger.error(f"Error during sync generation for {strategy_name}: {e}", exc_info=True)
        raw_output = f"[ERROR: {type(e).__name__}]"
        # Reset tokens on error
        llm._reset_token_counts()
        prompt_tokens = None
        completion_tokens = None
    time_taken = time.time() - t0
    # Return counts along with output and time
    return str(
        raw_output) if raw_output is not None else "[ERROR]", time_taken, prompt_tokens, completion_tokens


async def main():
    parser = argparse.ArgumentParser(description="Run Cogitator Benchmarks - Generation Phase")
    add_common_args(parser)
    add_generation_args(parser)
    args = parser.parse_args()

    config = load_and_merge_config(args, parser, config_section="generation")

    log_level = logging.DEBUG if config['debug'] else logging.INFO
    setup_logging(log_level)
    logger.info(f"Running Generation Phase with effective config: {config}")

    # --- Dataset Loading ---
    try:
        questions, golds = Datasets.load_dataset_by_name(config['dataset'], config['cutoff'])
    except Exception as e:
        logger.error(f"Failed to load dataset '{config['dataset']}': {e}", exc_info=True)
        sys.exit(1)

    # --- LLM Initialization ---
    try:
        key_from_cli = getattr(args, 'openai_key', None)
        env_var_name = config.get('openai_key_env_var', DEFAULT_OPENAI_ENV_VAR)
        key_from_env = os.getenv(env_var_name) if env_var_name else None
        openai_api_key = key_from_cli or key_from_env

        # Use the single LLM instance for all strategies
        llm = get_llm(
            provider=config['provider'],
            model_name=config['model_name'],
            openai_key=openai_api_key,
            ollama_host=config.get('ollama_host', None),
            llm_params=config.get('llm_params', {})  # Pass global LLM params
        )
    except Exception as e:
        logger.error(f"Failed to initialize LLM: {e}", exc_info=True)
        sys.exit(1)

    # --- Strategy Initialization ---
    logger.info("Initializing CoT strategies based on config...")
    instances = {}
    global_llm_params = config.get('llm_params', {})  # Re-get for clarity
    global_seed = global_llm_params.get('seed')
    global_max_tokens = global_llm_params.get('max_tokens')

    def extract_init_params(cfg: Dict[str, Any], keys: List[str]) -> Dict[str, Any]:
        """Helper to extract specific init parameters from strategy config."""
        return {k: cfg[k] for k in keys if k in cfg}

    # Initialize only enabled strategies
    if _is_strategy_enabled(config, "AutoCoT"):
        cfg = _get_strategy_config(config, "AutoCoT")
        params = extract_init_params(cfg,
                                     ["n_demos", "max_q_tokens", "max_steps", "prompt_template",
                                      "max_retries"])
        # Pass relevant global params if not overridden in strategy config
        params['max_tokens'] = cfg.get('max_tokens', global_max_tokens)
        params['rand_seed'] = cfg.get('seed', global_seed)  # Use 'rand_seed' for AutoCoT
        instances["AutoCoT"] = AutoCoT(llm, **params)
        logger.info(f"Initialized AutoCoT with params: {params}")

    if _is_strategy_enabled(config, "CDWCoT"):
        cfg = _get_strategy_config(config, "CDWCoT")
        params = extract_init_params(cfg, ["pool_size", "n_clusters", "lr", "sample_size",
                                           "max_grad_norm", "init_pool_retries"])
        params['max_tokens'] = cfg.get('max_tokens', global_max_tokens)
        params['seed'] = cfg.get('seed', global_seed)
        instances["CDWCoT"] = CDWCoT(llm, **params)
        logger.info(f"Initialized CDWCoT with params: {params}")

    if _is_strategy_enabled(config, "SelfConsistency"):
        cfg = _get_strategy_config(config, "SelfConsistency")
        params = extract_init_params(cfg, ["n_samples"])
        # Determine extraction format based on config hierarchy
        sc_extraction_format = cfg.get("internal_extraction_format")
        if sc_extraction_format is None:  # If not in strategy config, check global bench config
            sc_extraction_format = "json" if config.get('use_json_strategies',
                                                        False) else "heuristic"
        params['internal_extraction_format'] = sc_extraction_format
        # Pass relevant global/strategy params
        params['seed'] = cfg.get('seed', global_seed)
        params['max_tokens'] = cfg.get('max_tokens', global_max_tokens)
        params['temperature'] = cfg.get('temperature', global_llm_params.get(
            'temperature'))  # Explicitly get temperature
        params['stop'] = cfg.get('stop')  # Optional stop sequences
        instances["SelfConsistency"] = SelfConsistency(llm, **params)
        logger.info(f"Initialized SelfConsistency with params: {params}")

    if _is_strategy_enabled(config, "LeastToMost"):
        cfg = _get_strategy_config(config, "LeastToMost")
        params = extract_init_params(cfg, ["max_subqs", "decompose_prompt_template",
                                           "solve_prompt_template", "final_answer_prompt_template"])
        # Determine format
        ltm_format = cfg.get("intermediate_output_format")
        if ltm_format is None:
            ltm_format = "json" if config.get('use_json_strategies', False) else "text"
        params['intermediate_output_format'] = ltm_format
        # Pass relevant global/strategy params
        params['max_tokens'] = cfg.get('max_tokens', global_max_tokens)
        params['seed'] = cfg.get('seed', global_seed)
        instances["LeastToMost"] = LeastToMost(llm, **params)
        logger.info(f"Initialized LeastToMost with params: {params}")

    if _is_strategy_enabled(config, "TreeOfThoughts"):
        cfg = _get_strategy_config(config, "TreeOfThoughts")
        params = extract_init_params(cfg, ["max_depth", "num_branches", "sims", "c_puct",
                                           "expand_prompt", "eval_prompt"])
        params['max_tokens'] = cfg.get('max_tokens', global_max_tokens)
        params['seed'] = cfg.get('seed', global_seed)
        instances["TreeOfThoughts"] = TreeOfThoughts(llm, **params)
        logger.info(f"Initialized TreeOfThoughts with params: {params}")

    if _is_strategy_enabled(config, "GraphOfThoughts"):
        cfg = _get_strategy_config(config, "GraphOfThoughts")
        # Note: graph_of_operations is passed at runtime, not init
        params = extract_init_params(cfg, ["final_answer_format", "prompts"])
        # Determine format
        got_format = cfg.get("final_answer_format")
        if got_format is None:
            got_format = "json" if config.get('use_json_strategies', False) else "text"
        params['final_answer_format'] = got_format
        # Pass relevant global/strategy params
        params['max_tokens'] = cfg.get('max_tokens', global_max_tokens)
        params['seed'] = cfg.get('seed', global_seed)
        # Handle prompts merging if needed, default prompts are in GoT class
        if "prompts" in cfg:
            params["prompts"] = cfg["prompts"]

        instances["GraphOfThoughts"] = GraphOfThoughts(llm, **params)
        logger.info(f"Initialized GraphOfThoughts with params: {params}")
        # We still need to log the GoO from the config later in get_methods_to_run

    # --- Get Runnables ---
    all_methods_to_run = get_methods_to_run(config, instances, llm)
    # Filter out methods that failed initialization (func is None)
    runnable_methods = [(name, func) for name, func in all_methods_to_run if func is not None]
    methods_to_run_names = [name for name, func in runnable_methods]  # Names for setup phase

    if not runnable_methods:
        logger.error("No strategies are enabled or successfully initialized. Exiting.")
        sys.exit(1)

    # --- Setup Phase ---
    semaphore = asyncio.Semaphore(config['concurrency'])
    setup_successful = await run_setup_phase(config, llm, methods_to_run_names, questions, golds,
                                             semaphore, instances)

    # --- Generation Phase ---
    output_filepath = config['output_file']
    logger.info(
        f"Starting generation for {len(runnable_methods)} enabled methods, saving to {output_filepath}...")

    with open(output_filepath, 'w') as outfile:
        if config['use_async']:
            tasks = []
            task_info = []  # Store (index, question, method_name) for matching results
            for i, q in enumerate(questions):
                for name, model_func_wrapped in runnable_methods:
                    # Pass llm instance to run_single_async
                    tasks.append(run_single_async(q, model_func_wrapped, llm, semaphore))
                    task_info.append((i, q, name))

            logger.info(f"Running {len(tasks)} async generation tasks...")
            # results will be list of (raw_output, time_taken, p_tok, c_tok) tuples
            gen_outputs = await asyncio.gather(*tasks)
            logger.info("Async generation tasks complete.")

            # Process results and write to file
            for task_idx, (raw_output, time_taken, p_tok, c_tok) in enumerate(
                gen_outputs):  # Unpack 4
                i, q, name = task_info[task_idx]
                record = {
                    "question": q,
                    "gold": golds[i],
                    "method": name,
                    "dataset": config['dataset'],
                    "raw_output": raw_output,
                    "time": time_taken,
                    "prompt_tokens": p_tok,  # Add tokens (can be None)
                    "completion_tokens": c_tok  # Add tokens (can be None)
                }
                outfile.write(json.dumps(record) + '\n')
                # Log details including token counts
                token_log = f"P:{p_tok if p_tok is not None else '?'}, C:{c_tok if c_tok is not None else '?'}"
                logger.debug(f"Generated: Q{i + 1} - {name} ({time_taken:.2f}s, {token_log})")

        else:  # Synchronous loop
            for i, q in enumerate(questions):
                logger.info(f"Processing Question {i + 1}/{len(questions)}")
                for name, model_func_wrapped in runnable_methods:

                    # Check if setup was required and successful for relevant strategies
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
                    # Pass llm instance and name to run_single_sync
                    raw_output, time_taken, p_tok, c_tok = run_single_sync(q, model_func_wrapped,
                                                                           llm, name)  # Unpack 4
                    record = {
                        "question": q,
                        "gold": golds[i],
                        "method": name,
                        "dataset": config['dataset'],
                        "raw_output": raw_output,
                        "time": time_taken,
                        "prompt_tokens": p_tok,  # Add tokens
                        "completion_tokens": c_tok  # Add tokens
                    }
                    outfile.write(json.dumps(record) + '\n')
                    # Log details including token counts
                    token_log = f"P:{p_tok if p_tok is not None else '?'}, C:{c_tok if c_tok is not None else '?'}"
                    logger.debug(f"Generated: Q{i + 1} - {name} ({time_taken:.2f}s, {token_log})")

    logger.info(f"Generation complete. Raw results saved to {output_filepath}")


if __name__ == "__main__":
    asyncio.run(main())

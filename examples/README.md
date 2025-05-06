## Examples

| File                                                           | Description                                                                                                                   |
|----------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------|
| [run_simple_example.py](run_simple_example.py)                 | A simple end-to-end example of using the Cogitator library                                                                    |
| [run_least_to_most.py](run_least_to_most.py)                   | Example of using the Least-to-Most prompting strategy                                                                         |
| [run_sc_cot.py](run_sc_cot.py)                                 | Example of using the Self-Consistency prompting strategy                                                                      |
| [run_auto_cot.py](run_auto_cot.py)                             | Example of using the Automatic CoT prompting strategy                                                                         |
| [run_tree_of_thoughts.py](run_tree_of_thoughts.py)             | Example of using the Tree of Thoughts prompting framework                                                                     |
| [run_graph_of_thoughts.py](run_graph_of_thoughts.py)           | Example of using the Graph of Thoughts prompting framework                                                                    |
| [run_graph_of_thoughts_sort.py](run_graph_of_thoughts_sort.py) | Example of using the Graph of Thoughts for sorting a list of numbers (from [here](https://github.com/spcl/graph-of-thoughts)) |
| [run_cdw_cot.py](run_cdw_cot.py)                               | Example of using the Clustered Distance-Weighted CoT prompting framework                                                      |
| [shared.py](shared.py)                                         | Shared utilies and settings for the examples                                                                                  |

## Running Examples

```bash
# Run the Least-to-Most example (OpenAI)
python examples/run_least_to_most.py --provider openai --model-name gpt-4.1-nano
```

```bash
# Run the Self-Consistency example (Ollama)
python examples/run_least_to_most.py --provider ollama --model-name gemma3:4b
```

```bash
# Run all examples (Ollama)
make example-ollama
```

```bash
# Run all examples (OpenAI)
make example-openai
```

Note that the examples should be run from the root directory of the repository.
Additionally, to use `gemma3:4b` (or any other model like `gemma3:12b`) with Ollama, it must be pulled (or downloaded) first.

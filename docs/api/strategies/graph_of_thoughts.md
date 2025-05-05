# Graph of Thoughts Framework

Implementation of the Graph of Thoughts (GoT) reasoning framework, inspired by [this paper](https://arxiv.org/abs/2308.09687).

This implementation represents the reasoning process as a graph where nodes are thoughts and edges represent transformations. The
flow of reasoning is controlled by a **Graph of Operations (GoO)**.

## Defining the Graph of Operations (GoO)

To use `GraphOfThoughts`, you must provide a `graph_of_operations` argument to the `run_async` method. This argument is a list of
tuples, where each tuple defines an operation step:

`graph_of_operations: List[Tuple[str, Dict]]`

* The first element of the tuple is the **name** of the operation (e.g., `'Generate'`, `'Score'`, `'KeepBest'`).
* The second element is a **dictionary** containing the parameters specific to that operation.

**Example GoO:**

```python
from cogitator import ThoughtExpansion

EXAMPLE_GOO = [
    # Step 1: Generate 3 new thoughts from the initial question (in 'frontier' set)
    #         Store results in the 'generated_thoughts' set. Use the 'expand' prompt. Expect ThoughtExpansion schema.
    ('Generate', {'k': 3, 'target_set': 'frontier', 'output_set': 'generated_thoughts', 'prompt_key': 'expand', 'response_schema': ThoughtExpansion}),

    # Step 2: Score the thoughts generated in the previous step. Use 'evaluate' prompt.
    ('Score', {'target_set': 'generated_thoughts', 'prompt_key': 'evaluate'}),

    # Step 3: Keep only the single best-scoring thought from the previous step.
    #         Put the result back into the 'frontier' set for potential further steps or final answer generation.
    ('KeepBest', {'N': 1, 'target_set': 'generated_thoughts', 'output_set': 'frontier'})
]
```

## Main Class (`GraphOfThoughts`)

::: cogitator.strategies.graph_of_thoughts.GraphOfThoughts
    options:
        show_root_heading: true
        show_source: true
        members_order: source
        heading_level: 3

## Available Operations

Here are the standard operations available. You can create custom operations by subclassing `GoTOperation`.

### Base Operation Class

::: cogitator.strategies.graph_of_thoughts.GoTOperation
    options:
        show_root_heading: true
        show_source: false
        members_order: source
        heading_level: 3

### Generate Operation

::: cogitator.strategies.graph_of_thoughts.GenerateOp
    options:
        show_root_heading: true
        show_source: false
        members_order: source
        heading_level: 3

# Exclude inherited methods if desired

# members: ["__init__", "execute_async", "execute"] # Or list specific ones

### Score Operation

::: cogitator.strategies.graph_of_thoughts.ScoreOp
    options:
        show_root_heading: true
        show_source: false
        members_order: source
        heading_level: 3

### KeepBest Operation

::: cogitator.strategies.graph_of_thoughts.KeepBestOp
    options:
        show_root_heading: true
        show_source: false
        members_order: source
        heading_level: 3

### Aggregate Operation

::: cogitator.strategies.graph_of_thoughts.AggregateOp
    options:
        show_root_heading: true
        show_source: false
        members_order: source
        heading_level: 3

## Internal State (Advanced)

These classes manage the internal graph structure.

### GoTNode

::: cogitator.strategies.graph_of_thoughts.GoTNode
    options:
        show_root_heading: true
        show_source: false
        members: ["__init__"] # Only show init maybe?
        heading_level: 3

### GraphReasoningState

::: cogitator.strategies.graph_of_thoughts.GraphReasoningState
    options:
        show_root_heading: true
        show_source: false
        heading_level: 3

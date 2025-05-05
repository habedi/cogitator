"""Initializes the strategies subpackage.

Exports the main CoT strategy classes implemented within this subpackage, making them
available for direct import from `cogitator.strategies`. This includes various CoT and related reasoning frameworks.
"""

from .auto_cot import AutoCoT
from .cdw_cot import CDWCoT
from .graph_of_thoughts import GraphOfThoughts
from .least_to_most import LeastToMost
from .sc_cot import SelfConsistency
from .tree_of_thoughts import TreeOfThoughts

__all__ = [
    "AutoCoT",
    "CDWCoT",
    "GraphOfThoughts",
    "LeastToMost",
    "SelfConsistency",
    "TreeOfThoughts",
]

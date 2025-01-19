from enum import Enum


class Process(str, Enum):
    """
    Class representing the different processes that can be used to tackle tasks
    """

    sequential = "sequential"
    hierarchical = "hierarchical"
    consensual = "consensual"

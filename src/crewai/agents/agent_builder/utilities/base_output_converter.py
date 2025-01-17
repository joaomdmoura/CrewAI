from abc import ABC, abstractmethod
from typing import Any, Optional

from pydantic import BaseModel, Field


class OutputConverter(BaseModel, ABC):
    """
    Abstract base class for converting task results into structured formats.

    This class provides a framework for converting unstructured text into
    either Pydantic models or JSON, tailored for specific agent requirements.
    It uses a language model to interpret and structure the input text based
    on given instructions.

    Attributes:
        text (str): The input text to be converted.
        llm (Any): The language model used for conversion.
        model (Any): The target model for structuring the output.
        instructions (str): Specific instructions for the conversion process.
        max_attempts (int): Maximum number of conversion attempts (default: 3).
    """

    text: str = Field(description="Text to be converted.")
    llm: Any = Field(description="The language model to be used to convert the text.")
    model: Any = Field(description="The model to be used to convert the text.")
    instructions: str = Field(description="Conversion instructions to the LLM.")
    max_attempts: int = Field(
        description="Max number of attempts to try to get the output formatted.",
        default=3,
    )

    @abstractmethod
    def to_pydantic(self, current_attempt=1):
        """Convert text to pydantic."""
        pass

    @abstractmethod
    def to_json(self, current_attempt=1):
        """Convert text to json."""
        pass

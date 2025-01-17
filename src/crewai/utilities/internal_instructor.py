import warnings
from typing import Any, Optional, Type


class InternalInstructor:
    """Class that wraps an agent llm with instructor."""

    def __init__(
        self,
        content: str,
        model: Type,
        agent: Optional[Any] = None,
        llm: Optional[str] = None,
    ):
        self.content = content
        self.agent = agent
        self.llm = llm
        self.model = model
        self._client = None
        self.set_instructor()

    def set_instructor(self):
        """Set instructor."""
        if self.agent and not self.llm:
            self.llm = self.agent.function_calling_llm or self.agent.llm

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            import instructor
            from litellm import completion

            self._client = instructor.from_litellm(completion)

    def to_json(self):
        model = self.to_pydantic()
        return model.model_dump_json(indent=2)

    def to_pydantic(self):
        messages = [{"role": "user", "content": self.content}]
        model = self._client.chat.completions.create(
            model=self.llm.model, response_model=self.model, messages=messages
        )
        return model

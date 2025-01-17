import warnings
from typing import Any, Dict, Optional

from litellm.integrations.custom_logger import CustomLogger
from litellm.types.utils import Usage

from crewai.agents.agent_builder.utilities.base_token_process import TokenProcess


class TokenCalcHandler(CustomLogger):
    def __init__(self, token_cost_process: Optional[TokenProcess]):
        self.token_cost_process = token_cost_process

    def log_success_event(
        self,
        kwargs: Dict[str, Any],
        response_obj: Dict[str, Any],
        start_time: float,
        end_time: float,
    ) -> None:
        if self.token_cost_process is None:
            return

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            if isinstance(response_obj, dict) and "usage" in response_obj:
                usage: Usage = response_obj["usage"]
                if usage:
                    self.token_cost_process.sum_successful_requests(1)
                    if hasattr(usage, "prompt_tokens"):
                        self.token_cost_process.sum_prompt_tokens(usage.prompt_tokens)
                    if hasattr(usage, "completion_tokens"):
                        self.token_cost_process.sum_completion_tokens(usage.completion_tokens)
                    if hasattr(usage, "prompt_tokens_details") and usage.prompt_tokens_details:
                        self.token_cost_process.sum_cached_prompt_tokens(
                            usage.prompt_tokens_details.cached_tokens
                        )

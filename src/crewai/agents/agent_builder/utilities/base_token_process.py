from crewai.types.usage_metrics import UsageMetrics


class TokenProcess:
    def __init__(self):
        self.total_tokens: int = 0
        self.prompt_tokens: int = 0
        self.cached_prompt_tokens: int = 0
        self.completion_tokens: int = 0
        self.successful_requests: int = 0

    def sum_prompt_tokens(self, tokens: int):
        self.prompt_tokens = self.prompt_tokens + tokens
        self.total_tokens = self.total_tokens + tokens

    def sum_completion_tokens(self, tokens: int):
        self.completion_tokens = self.completion_tokens + tokens
        self.total_tokens = self.total_tokens + tokens

    def sum_cached_prompt_tokens(self, tokens: int):
        self.cached_prompt_tokens = self.cached_prompt_tokens + tokens

    def sum_successful_requests(self, requests: int):
        self.successful_requests = self.successful_requests + requests

    def get_summary(self) -> UsageMetrics:
        return UsageMetrics(
            total_tokens=self.total_tokens,
            prompt_tokens=self.prompt_tokens,
            cached_prompt_tokens=self.cached_prompt_tokens,
            completion_tokens=self.completion_tokens,
            successful_requests=self.successful_requests,
        )

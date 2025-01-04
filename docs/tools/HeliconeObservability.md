# Helicone

## Description

[Helicone](https://helicone.ai) is a open source developer tool that allows you to easily add caching, ETL with 1 line of code.

Get full visibility into:

- [Metrics](https://www.helicone.ai/) - ex: cost, latency, time to first token, etc...
- [Fine Tuning](https://docs.helicone.ai/features/advanced-usage/fine-tuning) - Easily export or your data to easily be fine tuned.
- [ETL](https://docs.helicone.ai/features/advanced-usage/etl) - Move data with any of our connectors (PostHog, Datadog, Segment)
- [Prompts](https://docs.helicone.ai/features/prompts) - Track your prompt versions auto-magically
- [Users](https://docs.helicone.ai/custom-parameters/user-tracking) - Keep track of users and limit their usage.
- [Traces](https://docs.helicone.ai/use-cases/segmentation) - Segment your data via traces and slice and dice your data in various ways.

## Installation

- Get an API key from [helicone.ai](https://helicone.ai) and set it in environment variables (`OPENAI_API_BASE`).

## Example

```python
os.environ["OPENAI_API_BASE"] = f"https://oai.helicone.ai/{HELICONE_API_KEY}/v1
```

That is it! Now you will get full visibility into your application.

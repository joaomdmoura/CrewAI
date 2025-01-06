import os
from unittest import mock

import pytest

from crewai.agent import Agent
from crewai.llm import LLM


def test_agent_with_custom_llm():
    """Test creating an agent with a custom LLM."""
    custom_llm = LLM(model="gpt-4")
    agent = Agent()
    agent.role = "test"
    agent.goal = "test"
    agent.backstory = "test"
    agent.llm = custom_llm
    agent.allow_delegation = False
    agent.post_init_setup()
    
    assert isinstance(agent.llm, LLM)
    assert agent.llm.model == "gpt-4"

def test_agent_with_uppercase_llm_param():
    """Test creating an agent with uppercase 'LLM' parameter."""
    custom_llm = LLM(model="gpt-4")
    with pytest.warns(DeprecationWarning):
        agent = Agent()
        agent.role = "test"
        agent.goal = "test"
        agent.backstory = "test"
        setattr(agent, 'LLM', custom_llm)  # Using uppercase LLM
        agent.allow_delegation = False
        agent.post_init_setup()
    
    assert isinstance(agent.llm, LLM)
    assert agent.llm.model == "gpt-4"
    assert not hasattr(agent, 'LLM')

def test_agent_llm_parameter_types():
    """Test LLM parameter type handling."""
    env_vars = {
        "temperature": "0.7",
        "max_tokens": "100",
        "presence_penalty": "0.5",
        "logprobs": "true",
        "logit_bias": '{"50256": -100}',
        "callbacks": "callback1,callback2",
    }
    with mock.patch.dict(os.environ, env_vars):
        agent = Agent()
        agent.role = "test"
        agent.goal = "test"
        agent.backstory = "test"
        agent.llm = "gpt-4"
        agent.allow_delegation = False
        agent.post_init_setup()
        
        assert isinstance(agent.llm, LLM)
        assert agent.llm.temperature == 0.7
        assert agent.llm.max_tokens == 100
        assert agent.llm.presence_penalty == 0.5
        assert agent.llm.logprobs is True
        assert agent.llm.logit_bias == {50256: -100.0}
        assert agent.llm.callbacks == ["callback1", "callback2"]

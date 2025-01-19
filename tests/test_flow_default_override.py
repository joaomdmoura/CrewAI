"""Test that persisted state properly overrides default values."""

import os
from typing import Optional

import pytest
from pydantic import BaseModel

from crewai.flow.flow import Flow, FlowState, start
from crewai.flow.persistence import persist
from crewai.flow.persistence.sqlite import SQLiteFlowPersistence


class PoemState(FlowState):
    """Test state model with default values that should be overridden."""
    sentence_count: int = 1000  # Default that should be overridden
    has_set_count: bool = False  # Track whether we've set the count


def test_default_value_override():
    """Test that persisted state values override class defaults."""

    @persist()
    class PoemFlow(Flow[PoemState]):
        initial_state = PoemState

        @start()
        def set_sentence_count(self):
            print("Setting sentence count")
            print(self.state)
            # Only set sentence_count on first run, not when loading from persistence
            if self.state.has_set_count and self.state.sentence_count == 2:
                self.state.sentence_count = 3
            elif self.state.has_set_count and self.state.sentence_count == 1000:
                self.state.sentence_count = 1000
            elif self.state.has_set_count and self.state.sentence_count == 5:
                self.state.sentence_count = 5
            else:
                self.state.sentence_count = 2
                self.state.has_set_count = True

    # First run - should set sentence_count to 2
    flow1 = PoemFlow()
    flow1.kickoff()
    original_uuid = flow1.state.id
    assert flow1.state.sentence_count == 2

    # Second run - should load sentence_count=2 instead of default 1000
    flow2 = PoemFlow()
    flow2.kickoff(inputs={"id": original_uuid})
    assert flow2.state.sentence_count == 3  # Should load 2, not default 1000

    # Third run - should not load sentence_count=2 instead of default 1000
    flow2 = PoemFlow()
    flow2.kickoff(inputs={"has_set_count": True})
    assert flow2.state.sentence_count == 1000  # Should load 1000, not 2

    # Third run - explicit override should work
    flow3 = PoemFlow(
        id=original_uuid,
        sentence_count=5,  # Override persisted value
    )
    assert flow3.state.sentence_count == 5  # Should use override value

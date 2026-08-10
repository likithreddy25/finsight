"""Unit test for FinSightAgent's tool-calling loop, mocking the OpenAI client and the
pipeline dispatch so this runs offline (no SEC EDGAR / OpenAI network calls needed).

It exercises a realistic multi-turn trace: the model asks for a health score, then a
follow-up filing search, then returns a final synthesized answer -- verifying that tool
results get threaded back into the message list correctly and the trace is recorded
in order.

Run: pytest tests/test_agent.py
"""
from __future__ import annotations

import json
import types
from dataclasses import dataclass
from unittest.mock import MagicMock

from agent.analyst import FinSightAgent


@dataclass
class _FakeFunction:
    name: str
    arguments: str


@dataclass
class _FakeToolCall:
    id: str
    function: _FakeFunction


class _FakeMessage:
    def __init__(self, content=None, tool_calls=None):
        self.content = content
        self.tool_calls = tool_calls

    def model_dump(self, exclude_unset=True):
        d = {"role": "assistant", "content": self.content}
        if self.tool_calls:
            d["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                }
                for tc in self.tool_calls
            ]
        return d


def _response(message: _FakeMessage):
    return types.SimpleNamespace(choices=[types.SimpleNamespace(message=message)])


def test_agent_multi_turn_tool_loop():
    turn1 = _response(
        _FakeMessage(
            tool_calls=[_FakeToolCall("call_1", _FakeFunction("get_health_score", json.dumps({"ticker": "AAPL"})))]
        )
    )
    turn2 = _response(
        _FakeMessage(
            tool_calls=[
                _FakeToolCall(
                    "call_2",
                    _FakeFunction("query_filings", json.dumps({"question": "supply chain risk", "ticker": "AAPL"})),
                )
            ]
        )
    )
    turn3 = _response(_FakeMessage(content="AAPL scored 7.2/10; filings show moderate supply-chain risk language."))

    fake_client = MagicMock()
    fake_client.chat.completions.create.side_effect = [turn1, turn2, turn3]

    agent = FinSightAgent.__new__(FinSightAgent)  # bypass __init__'s real OpenAI/pipeline construction
    agent.client = fake_client
    agent.model = "gpt-4o-mini"
    agent.max_turns = 6

    call_log = []

    def fake_dispatch(name, arguments):
        call_log.append((name, arguments))
        if name == "get_health_score":
            return {"ticker": "AAPL", "score": 7.2, "rationale": "fake"}
        if name == "query_filings":
            return {"results": [{"ticker": "AAPL", "form_type": "10-K", "text": "supply chain risk excerpt..."}]}
        raise AssertionError(f"unexpected tool {name}")

    agent._dispatch = fake_dispatch

    result = agent.run("How healthy is AAPL and any supply chain risk?")

    assert call_log == [
        ("get_health_score", {"ticker": "AAPL"}),
        ("query_filings", {"question": "supply chain risk", "ticker": "AAPL"}),
    ]
    assert result.answer == "AAPL scored 7.2/10; filings show moderate supply-chain risk language."
    assert [t.tool for t in result.trace] == ["get_health_score", "query_filings"]
    assert fake_client.chat.completions.create.call_count == 3


def test_agent_stops_immediately_when_no_tool_call_needed():
    turn1 = _response(_FakeMessage(content="I need a ticker symbol to look anything up."))
    fake_client = MagicMock()
    fake_client.chat.completions.create.side_effect = [turn1]

    agent = FinSightAgent.__new__(FinSightAgent)
    agent.client = fake_client
    agent.model = "gpt-4o-mini"
    agent.max_turns = 6
    agent._dispatch = MagicMock(side_effect=AssertionError("should not be called"))

    result = agent.run("What's a good stock?")

    assert result.answer == "I need a ticker symbol to look anything up."
    assert result.trace == []
    assert fake_client.chat.completions.create.call_count == 1

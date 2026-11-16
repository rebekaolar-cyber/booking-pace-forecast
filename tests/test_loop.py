"""Tests for the agent loop with a scripted fake client (no API key, no network)."""
import copy
import json
from types import SimpleNamespace

import pytest

from agent.contract import TOOL_SCHEMAS
from agent.fake_backend import FakeBackend
import anthropic

from agent.loop import (API_PROBLEM, CUT_OFF, INTERNAL_PROBLEM, answer, block_to_dict,
                        load_prompt)


# --------------------------------------------------------------------------- fakes

def text(t):
    return SimpleNamespace(type="text", text=t)


def tool_use(id_, name, input_=None):
    return SimpleNamespace(type="tool_use", id=id_, name=name, input=input_ or {})


def response(blocks, stop_reason, input_tokens=10, output_tokens=5):
    return SimpleNamespace(content=blocks, stop_reason=stop_reason,
                           usage=SimpleNamespace(input_tokens=input_tokens,
                                                 output_tokens=output_tokens))


class FakeAPIError(anthropic.APIError):
    """A real SDK error class, built without an HTTP request."""

    def __init__(self, message):
        Exception.__init__(self, message)


class FakeClient:
    """Mimics `client.messages.create`: returns scripted responses, records each request."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.requests = []
        self.messages = self

    def create(self, **kwargs):
        self.requests.append(copy.deepcopy(kwargs))  # the loop keeps mutating `messages`
        item = self.responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def run(responses, **kwargs):
    client = FakeClient(responses)
    return answer("How busy is Saturday?", FakeBackend(), client=client, **kwargs), client


# --------------------------------------------------------------------------- tests

def test_direct_answer():
    result, client = run([response([text("Hello "), text("there.")], "end_turn")])
    assert result.answer == "Hello there."
    assert result.stop_reason == "answered"
    assert result.tool_calls == []
    assert result.error is None
    assert len(client.requests) == 1


def test_request_uses_prompt_tools_and_settings():
    _, client = run([response([text("ok")], "end_turn")], model="m-test", max_tokens=123)
    request = client.requests[0]
    assert request["system"] == load_prompt("v2")  # the default prompt
    assert request["tools"] == TOOL_SCHEMAS
    assert request["model"] == "m-test"
    assert request["max_tokens"] == 123
    assert request["messages"] == [{"role": "user", "content": "How busy is Saturday?"}]


def test_prompt_file_is_loaded():
    prompt = load_prompt("v1")
    assert "get_today_and_data_range" in prompt
    assert "Based on:" in prompt


def test_one_tool_round_then_answer():
    result, client = run([
        response([text("Let me check."), tool_use("tu_1", "get_today_and_data_range")],
                 "tool_use"),
        response([text("Today is 2017-06-01.")], "end_turn"),
    ])
    assert result.answer == "Today is 2017-06-01."
    assert result.stop_reason == "answered"
    assert [c.name for c in result.tool_calls] == ["get_today_and_data_range"]
    assert result.tool_calls[0].output["today"] == "2017-06-01"

    messages = client.requests[1]["messages"]
    assert messages[1] == {"role": "assistant", "content": [
        {"type": "text", "text": "Let me check."},
        {"type": "tool_use", "id": "tu_1", "name": "get_today_and_data_range", "input": {}},
    ]}
    (tool_result,) = messages[2]["content"]
    assert messages[2]["role"] == "user"
    assert tool_result["type"] == "tool_result"
    assert tool_result["tool_use_id"] == "tu_1"
    assert json.loads(tool_result["content"])["today"] == "2017-06-01"
    assert "is_error" not in tool_result
    json.dumps(messages)  # everything sent back must be plain JSON


def test_several_tools_in_one_turn_go_back_in_one_message():
    result, client = run([
        response([tool_use("a", "get_accuracy"),
                  tool_use("b", "get_staffing", {"date": "2017-06-03", "role": "bar"})],
                 "tool_use"),
        response([text("done")], "end_turn"),
    ])
    assert len(result.tool_calls) == 2
    results = client.requests[1]["messages"][2]["content"]
    assert [r["tool_use_id"] for r in results] == ["a", "b"]


def test_tool_error_sets_is_error():
    result, client = run([
        response([tool_use("tu_1", "get_forecast",
                           {"start_date": "2017-01-01", "end_date": "2017-01-02"})], "tool_use"),
        response([text("That date is in the past.")], "end_turn"),
    ])
    tool_result = client.requests[1]["messages"][2]["content"][0]
    assert tool_result["is_error"] is True
    assert "past" in json.loads(tool_result["content"])["error"]
    assert result.stop_reason == "answered"


def test_round_limit_stops_a_model_that_keeps_calling_tools():
    looping = [response([tool_use(f"t{i}", "get_accuracy")], "tool_use") for i in range(10)]
    result, client = run(looping, max_rounds=4)
    assert result.stop_reason == "tool_limit"
    assert len(client.requests) == 4
    assert len(result.tool_calls) == 3  # the 4th round's call is never run: no round left to answer
    assert "too many steps" in result.answer


def test_tool_budget_runs_what_fits_and_refuses_the_rest():
    two_at_once = response([tool_use("a", "get_accuracy"), tool_use("b", "get_accuracy")],
                           "tool_use")
    result, client = run([two_at_once, copy.deepcopy(two_at_once),
                          response([text("Here is what I found.")], "end_turn")],
                         max_tool_calls=3)
    assert result.stop_reason == "answered"
    assert len(result.tool_calls) == 3  # only calls that really ran are recorded
    second_results = client.requests[2]["messages"][4]["content"]
    assert "error" not in json.loads(second_results[0]["content"])
    assert json.loads(second_results[1]["content"]) == {"error": "tool budget reached"}
    assert second_results[1]["is_error"] is True


def test_tool_budget_reached_on_every_round_still_ends():
    looping = [response([tool_use(f"t{i}", "get_accuracy")], "tool_use") for i in range(10)]
    result, client = run(looping, max_tool_calls=2)
    assert result.stop_reason == "tool_limit"
    assert len(client.requests) == 6  # default max_rounds
    assert len(result.tool_calls) == 2


def test_bar_staff_each_day_next_week_fits_the_default_budget():
    days = ["2017-06-05", "2017-06-06", "2017-06-07", "2017-06-08", "2017-06-09",
            "2017-06-10", "2017-06-11"]  # Mon..Sun
    week = [tool_use(f"s{i}", "get_staffing", {"date": d, "role": "bar"})
            for i, d in enumerate(days)]
    result, client = run([
        response([tool_use("t0", "get_today_and_data_range")], "tool_use"),
        response(week, "tool_use"),
        response([text("Bar staff per day: ...")], "end_turn"),
    ])
    assert len(week) == 7
    assert result.stop_reason == "answered"
    assert len(result.tool_calls) == 8
    assert all("error" not in c.output for c in result.tool_calls)


def test_tool_use_stop_without_tool_blocks_ends_instead_of_looping():
    result, client = run([response([text("I will check.")], "tool_use")] * 20)
    assert len(client.requests) == 1
    assert result.stop_reason == "error"
    assert result.answer == "I will check."
    assert "tool_use" in result.error


@pytest.mark.parametrize("stop, expected", [
    ("refusal", "refused"), ("pause_turn", "error"), ("something_new", "error")])
def test_other_stop_reasons_are_not_recorded_as_answered(stop, expected):
    result, _ = run([response([], stop)])
    assert result.stop_reason == expected
    assert result.answer  # still a friendly sentence


def test_non_dict_tool_input_becomes_a_validation_error_not_a_crash():
    result, client = run([
        response([tool_use("t1", "get_staffing", "2017-06-03")], "tool_use"),
        response([text("Which date did you mean?")], "end_turn"),
    ])
    assert result.stop_reason == "answered"
    tool_result = client.requests[1]["messages"][2]["content"][0]
    assert tool_result["is_error"] is True
    assert "date" in json.loads(tool_result["content"])["error"]


def test_thinking_blocks_are_ignored_in_text_but_echoed_back():
    thinking = SimpleNamespace(type="thinking", thinking="hmm", signature="sig")
    result, client = run([
        response([thinking, tool_use("t1", "get_accuracy")], "tool_use"),
        response([thinking, text("Accurate enough.")], "end_turn"),
    ])
    assert result.answer == "Accurate enough."
    assert client.requests[1]["messages"][1]["content"][0] == {
        "type": "thinking", "thinking": "hmm", "signature": "sig"}


def test_missing_usage_counts_as_zero_tokens():
    reply = SimpleNamespace(content=[text("ok")], stop_reason="end_turn")
    result, _ = run([reply])
    assert result.stop_reason == "answered"
    assert (result.input_tokens, result.output_tokens) == (0, 0)


def test_history_is_sent_as_plain_text_turns():
    history = [{"question": f"q{i}", "answer": f"a{i}"} for i in range(6)]
    history.append({"question": "failed", "answer": ""})  # empty text would be rejected
    client = FakeClient([response([text("Sunday: 80 guests.")], "end_turn")])
    answer("and Sunday?", FakeBackend(), client=client, history=history)
    messages = client.requests[0]["messages"]
    assert messages[0] == {"role": "user", "content": "q3"}  # only the last 4 turns
    assert messages[1] == {"role": "assistant", "content": "a3"}
    assert messages[-1] == {"role": "user", "content": "and Sunday?"}
    assert len(messages) == 2 * 3 + 1  # q3..q5 pairs; the empty one is skipped


def test_api_exception_returns_error_result():
    result, _ = run([FakeAPIError("overloaded")])
    assert result.stop_reason == "error"
    assert result.error == "overloaded"
    assert result.answer == API_PROBLEM


def test_api_exception_after_a_tool_round_keeps_the_tool_calls():
    result, _ = run([response([tool_use("t1", "get_accuracy")], "tool_use"),
                     FakeAPIError("timeout")])
    assert result.stop_reason == "error"
    assert result.answer == API_PROBLEM
    assert len(result.tool_calls) == 1


def test_code_bug_is_not_blamed_on_the_ai_service():
    result, _ = run([KeyError("oops")])
    assert result.stop_reason == "error"
    assert result.answer == INTERNAL_PROBLEM
    assert "KeyError" in result.error


def test_missing_api_key_without_client_returns_error(monkeypatch):
    for var in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN"):
        monkeypatch.delenv(var, raising=False)
    result = answer("hi", FakeBackend())
    assert result.stop_reason == "error"
    assert result.error


def test_tokens_are_summed_and_time_recorded():
    result, _ = run([
        response([tool_use("t1", "get_accuracy")], "tool_use", 100, 20),
        response([text("ok")], "end_turn", 150, 30),
    ])
    assert (result.input_tokens, result.output_tokens) == (250, 50)
    assert result.seconds >= 0
    assert result.tool_calls[0].seconds >= 0


def test_max_tokens_stop_is_recorded_as_truncated():
    result, _ = run([response([text("Saturday looks")], "max_tokens")])
    assert result.stop_reason == "truncated"
    assert result.answer == f"Saturday looks {CUT_OFF}"


def test_max_tokens_after_a_full_sentence_adds_no_note():
    result, _ = run([response([text("Saturday looks busy.")], "max_tokens")])
    assert result.stop_reason == "truncated"
    assert result.answer == "Saturday looks busy."


def test_max_tokens_with_no_text_says_it_was_cut_off():
    result, _ = run([response([], "max_tokens")])
    assert result.answer == CUT_OFF


def test_default_max_tokens_is_700():
    _, client = run([response([text("ok")], "end_turn")])
    assert client.requests[0]["max_tokens"] == 700


def test_result_records_model_and_prompt_version():
    result, _ = run([response([text("ok")], "end_turn")], model="m-x", prompt_version="v1")
    assert (result.model, result.prompt_version, result.question) == (
        "m-x", "v1", "How busy is Saturday?")


def test_block_to_dict_uses_model_dump_for_sdk_objects():
    from anthropic.types import ToolUseBlock
    block = ToolUseBlock(type="tool_use", id="t1", name="get_accuracy", input={})
    assert block_to_dict(block) == {"type": "tool_use", "id": "t1",
                                    "name": "get_accuracy", "input": {}}


def test_block_to_dict_rejects_unknown_types():
    with pytest.raises(ValueError):
        block_to_dict(SimpleNamespace(type="image"))


def test_prompt_v2_is_v1_plus_two_rules():
    """v2 adds only two rules (exact percentages; answer in the question's language), so a
    v1 -> v2 eval comparison isolates them."""
    v1, v2 = load_prompt("v1").splitlines(), load_prompt("v2").splitlines()
    added = [line for line in v2 if line not in v1]
    assert [line for line in v1 if line not in v2] == []
    assert len(added) == 2
    assert "14.5% below" in added[0] and "never rounded" in added[0]
    assert "language the question is written in" in added[1] and "English" in added[1]

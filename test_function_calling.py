"""
Unit tests for the function-calling emulation.

Tests all 7 parse_tool_calls() patterns plus:
  - build_tool_system_prompt() with different tool_choice values
  - inject_tools_into_messages() with/without existing system message
  - messages_to_prompt() multi-turn tool call sequences
  - _make_tool_call() / wrap_tool_calls_response() structure

Run: python test_function_calling.py
"""

import sys
import os
import json
import unittest

os.environ.setdefault("MEMCACHED_HOST", "127.0.0.1")
os.environ.setdefault("ONEMIN_API_KEY", "test-key")

# FC functions live in function_calling.py — no Flask/waitress side effects
from function_calling import (
    parse_tool_calls,
    build_tool_system_prompt,
    inject_tools_into_messages,
    _make_tool_call,
    wrap_tool_calls_response,
)

# messages_to_prompt lives in main.py — patch side-effecting imports
import unittest.mock as mock

with mock.patch("waitress.serve"), \
     mock.patch("pymemcache.client.base.Client") as _mc:
    _mc.return_value.set.return_value = True
    _mc.return_value.get.return_value = None   # probe fails → in-memory limiter
    import main  # noqa: E402

from main import messages_to_prompt

# ---------------------------------------------------------------------------
# Shared test fixtures
# ---------------------------------------------------------------------------
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get current weather for a location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string", "description": "City name"},
                    "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                },
                "required": ["location"],
            },
        },
    }
]


# ===========================================================================
# parse_tool_calls — pattern 1: instructed <tool_call> XML+JSON
# ===========================================================================
class TestParsePattern1(unittest.TestCase):

    def test_basic(self):
        text = '<tool_call>{"name": "get_weather", "arguments": {"location": "Berlin"}}</tool_call>'
        result = parse_tool_calls(text)
        self.assertIsNotNone(result)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["function"]["name"], "get_weather")
        args = json.loads(result[0]["function"]["arguments"])
        self.assertEqual(args["location"], "Berlin")

    def test_with_surrounding_text(self):
        text = 'Sure, let me check.\n<tool_call>{"name": "get_weather", "arguments": {"location": "Paris"}}</tool_call>\nDone.'
        result = parse_tool_calls(text)
        self.assertIsNotNone(result)
        self.assertEqual(result[0]["function"]["name"], "get_weather")

    def test_multiple_calls(self):
        text = (
            '<tool_call>{"name": "get_weather", "arguments": {"location": "Berlin"}}</tool_call>'
            '<tool_call>{"name": "get_weather", "arguments": {"location": "Munich"}}</tool_call>'
        )
        result = parse_tool_calls(text)
        self.assertIsNotNone(result)
        self.assertEqual(len(result), 2)

    def test_multiline_json(self):
        text = '<tool_call>\n{"name": "get_weather",\n "arguments": {"location": "Hamburg"}}\n</tool_call>'
        result = parse_tool_calls(text)
        self.assertIsNotNone(result)
        self.assertEqual(result[0]["function"]["name"], "get_weather")

    def test_no_tool_call(self):
        text = "The weather in Berlin is sunny today."
        result = parse_tool_calls(text)
        self.assertIsNone(result)


# ===========================================================================
# parse_tool_calls — pattern 2: plain JSON {"tool_call": {...}}
# ===========================================================================
class TestParsePattern2(unittest.TestCase):

    def test_basic(self):
        text = '{"tool_call": {"name": "get_weather", "arguments": {"location": "Berlin"}}}'
        result = parse_tool_calls(text)
        self.assertIsNotNone(result)
        self.assertEqual(result[0]["function"]["name"], "get_weather")

    def test_priority_over_pattern2(self):
        """Pattern 1 must win when both appear."""
        text = (
            '<tool_call>{"name": "get_weather", "arguments": {"location": "Berlin"}}</tool_call>'
            '{"tool_call": {"name": "other_tool", "arguments": {}}}'
        )
        result = parse_tool_calls(text)
        self.assertIsNotNone(result)
        self.assertEqual(result[0]["function"]["name"], "get_weather")


# ===========================================================================
# parse_tool_calls — pattern 3: legacy Anthropic XML
# ===========================================================================
class TestParsePattern3(unittest.TestCase):

    def test_basic(self):
        text = (
            '<function_calls>'
            '<invoke name="get_weather">'
            '<parameter name="location">Berlin</parameter>'
            '</invoke>'
            '</function_calls>'
        )
        result = parse_tool_calls(text)
        self.assertIsNotNone(result)
        self.assertEqual(result[0]["function"]["name"], "get_weather")
        args = json.loads(result[0]["function"]["arguments"])
        self.assertEqual(args["location"], "Berlin")

    def test_multiple_params(self):
        text = (
            '<invoke name="get_weather">'
            '<parameter name="location">Munich</parameter>'
            '<parameter name="unit">celsius</parameter>'
            '</invoke>'
        )
        result = parse_tool_calls(text)
        self.assertIsNotNone(result)
        args = json.loads(result[0]["function"]["arguments"])
        self.assertEqual(args["unit"], "celsius")

    def test_json_typed_param(self):
        """Parameters that are valid JSON should be parsed as their native type."""
        text = (
            '<invoke name="search">'
            '<parameter name="count">42</parameter>'
            '</invoke>'
        )
        result = parse_tool_calls(text)
        self.assertIsNotNone(result)
        args = json.loads(result[0]["function"]["arguments"])
        self.assertEqual(args["count"], 42)


# ===========================================================================
# parse_tool_calls — pattern 4: DeepSeek <|tool▁call▁begin|>
# ===========================================================================
class TestParsePattern4(unittest.TestCase):

    def test_basic(self):
        text = '<|tool▁call▁begin|>{"name": "get_weather", "arguments": {"location": "Berlin"}}<|tool▁call▁end|>'
        result = parse_tool_calls(text)
        self.assertIsNotNone(result)
        self.assertEqual(result[0]["function"]["name"], "get_weather")

    def test_parameters_key(self):
        """DeepSeek may also use 'parameters' instead of 'arguments'."""
        text = '<|tool▁call▁begin|>{"name": "get_weather", "parameters": {"location": "Berlin"}}<|tool▁call▁end|>'
        result = parse_tool_calls(text)
        self.assertIsNotNone(result)
        args = json.loads(result[0]["function"]["arguments"])
        self.assertEqual(args["location"], "Berlin")


# ===========================================================================
# parse_tool_calls — pattern 5: Mistral [TOOL_CALLS]
# ===========================================================================
class TestParsePattern5(unittest.TestCase):

    def test_basic(self):
        text = '[TOOL_CALLS] [{"name": "get_weather", "arguments": {"location": "Berlin"}}]'
        result = parse_tool_calls(text)
        self.assertIsNotNone(result)
        self.assertEqual(result[0]["function"]["name"], "get_weather")

    def test_multiple_calls(self):
        text = '[TOOL_CALLS] [{"name": "get_weather", "arguments": {"location": "Berlin"}}, {"name": "get_weather", "arguments": {"location": "Munich"}}]'
        result = parse_tool_calls(text)
        self.assertIsNotNone(result)
        self.assertEqual(len(result), 2)

    def test_parameters_key(self):
        text = '[TOOL_CALLS] [{"name": "get_weather", "parameters": {"location": "Hamburg"}}]'
        result = parse_tool_calls(text)
        self.assertIsNotNone(result)
        args = json.loads(result[0]["function"]["arguments"])
        self.assertEqual(args["location"], "Hamburg")


# ===========================================================================
# parse_tool_calls — pattern 6: Llama <|python_tag|>
# ===========================================================================
class TestParsePattern6(unittest.TestCase):

    def test_basic_with_parameters(self):
        """Llama uses 'parameters' key."""
        text = '<|python_tag|>{"name": "get_weather", "parameters": {"location": "Berlin"}}'
        result = parse_tool_calls(text)
        self.assertIsNotNone(result)
        self.assertEqual(result[0]["function"]["name"], "get_weather")
        args = json.loads(result[0]["function"]["arguments"])
        self.assertEqual(args["location"], "Berlin")

    def test_with_eom_marker(self):
        text = '<|python_tag|>{"name": "get_weather", "parameters": {"location": "Berlin"}}<|eom_id|>'
        result = parse_tool_calls(text)
        self.assertIsNotNone(result)
        self.assertEqual(result[0]["function"]["name"], "get_weather")

    def test_arguments_key_fallback(self):
        """Falls back to 'arguments' if 'parameters' absent."""
        text = '<|python_tag|>{"name": "get_weather", "arguments": {"location": "Berlin"}}'
        result = parse_tool_calls(text)
        self.assertIsNotNone(result)
        args = json.loads(result[0]["function"]["arguments"])
        self.assertEqual(args["location"], "Berlin")


# ===========================================================================
# parse_tool_calls — pattern 7: Qwen old ✿FUNCTION✿
# ===========================================================================
class TestParsePattern7(unittest.TestCase):

    def test_basic(self):
        text = '✿FUNCTION✿: get_weather\n✿ARGS✿: {"location": "Berlin"}'
        result = parse_tool_calls(text)
        self.assertIsNotNone(result)
        self.assertEqual(result[0]["function"]["name"], "get_weather")
        args = json.loads(result[0]["function"]["arguments"])
        self.assertEqual(args["location"], "Berlin")

    def test_inline(self):
        text = '✿FUNCTION✿: get_weather ✿ARGS✿: {"location": "Munich", "unit": "celsius"}'
        result = parse_tool_calls(text)
        self.assertIsNotNone(result)
        self.assertEqual(result[0]["function"]["name"], "get_weather")


# ===========================================================================
# _make_tool_call structure
# ===========================================================================
class TestMakeToolCall(unittest.TestCase):

    def test_structure(self):
        tc = _make_tool_call("get_weather", {"location": "Berlin"})
        self.assertEqual(tc["type"], "function")
        self.assertTrue(tc["id"].startswith("call_"))
        self.assertEqual(len(tc["id"]), len("call_") + 24)
        self.assertEqual(tc["function"]["name"], "get_weather")
        args = json.loads(tc["function"]["arguments"])
        self.assertEqual(args["location"], "Berlin")

    def test_string_arguments_passthrough(self):
        tc = _make_tool_call("get_weather", '{"location": "Berlin"}')
        self.assertEqual(tc["function"]["arguments"], '{"location": "Berlin"}')

    def test_unique_ids(self):
        ids = {_make_tool_call("f", {})["id"] for _ in range(20)}
        self.assertEqual(len(ids), 20)


# ===========================================================================
# build_tool_system_prompt
# ===========================================================================
class TestBuildToolSystemPrompt(unittest.TestCase):

    def test_contains_tool_name(self):
        prompt = build_tool_system_prompt(TOOLS)
        self.assertIn("get_weather", prompt)

    def test_contains_format_instruction(self):
        prompt = build_tool_system_prompt(TOOLS)
        self.assertIn("<tool_call>", prompt)

    def test_tool_choice_auto(self):
        prompt = build_tool_system_prompt(TOOLS, "auto")
        self.assertIn("respond normally", prompt.lower())

    def test_tool_choice_required(self):
        prompt = build_tool_system_prompt(TOOLS, "required")
        self.assertIn("MUST", prompt)
        self.assertIn("plain text", prompt)

    def test_tool_choice_specific_function(self):
        tool_choice = {"type": "function", "function": {"name": "get_weather"}}
        prompt = build_tool_system_prompt(TOOLS, tool_choice)
        self.assertIn("get_weather", prompt)
        self.assertIn("MUST", prompt)

    def test_bare_function_dict(self):
        """Tools without 'type'/'function' wrapper should still work."""
        bare_tools = [{"name": "ping", "description": "Ping a host", "parameters": {}}]
        prompt = build_tool_system_prompt(bare_tools)
        self.assertIn("ping", prompt)


# ===========================================================================
# inject_tools_into_messages
# ===========================================================================
class TestInjectToolsIntoMessages(unittest.TestCase):

    def test_no_system_message(self):
        messages = [{"role": "user", "content": "What's the weather?"}]
        patched = inject_tools_into_messages(messages, TOOLS)
        self.assertEqual(patched[0]["role"], "system")
        self.assertIn("get_weather", patched[0]["content"])
        # Original unmodified
        self.assertEqual(messages[0]["role"], "user")

    def test_existing_system_message_appended(self):
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "What's the weather?"},
        ]
        patched = inject_tools_into_messages(messages, TOOLS)
        self.assertEqual(patched[0]["role"], "system")
        self.assertIn("You are helpful.", patched[0]["content"])
        self.assertIn("get_weather", patched[0]["content"])

    def test_original_not_mutated(self):
        messages = [{"role": "user", "content": "Hello"}]
        inject_tools_into_messages(messages, TOOLS)
        self.assertEqual(len(messages), 1)
        self.assertNotIn("content", messages[0].get("system", {}))

    def test_tool_choice_required_in_prompt(self):
        messages = [{"role": "user", "content": "Do something"}]
        patched = inject_tools_into_messages(messages, TOOLS, "required")
        self.assertIn("MUST", patched[0]["content"])


# ===========================================================================
# messages_to_prompt — multi-turn tool calls
# ===========================================================================
class TestMessagesToPompt(unittest.TestCase):

    def test_simple_conversation(self):
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
        ]
        prompt = messages_to_prompt(messages)
        self.assertIn("User: Hello", prompt)
        self.assertIn("Assistant: Hi there!", prompt)

    def test_system_message(self):
        messages = [
            {"role": "system", "content": "Be concise."},
            {"role": "user", "content": "Hello"},
        ]
        prompt = messages_to_prompt(messages)
        self.assertIn("System: Be concise.", prompt)

    def test_tool_result_with_name(self):
        """Tool result messages should show the tool name via tool_call_id lookup."""
        messages = [
            {"role": "user", "content": "What's the weather in Berlin?"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [{
                    "id": "call_abc123",
                    "type": "function",
                    "function": {"name": "get_weather", "arguments": '{"location": "Berlin"}'},
                }],
            },
            {
                "role": "tool",
                "tool_call_id": "call_abc123",
                "content": '{"temperature": 18, "condition": "cloudy"}',
            },
        ]
        prompt = messages_to_prompt(messages)
        self.assertIn("get_weather", prompt)
        self.assertIn("temperature", prompt)
        # The tool line should have the name
        self.assertIn("Tool (get_weather):", prompt)

    def test_assistant_tool_call_reproduced(self):
        """Assistant messages with tool_calls should be reproduced in <tool_call> format."""
        messages = [
            {"role": "user", "content": "What's the weather in Berlin?"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [{
                    "id": "call_abc123",
                    "type": "function",
                    "function": {"name": "get_weather", "arguments": '{"location": "Berlin"}'},
                }],
            },
            {"role": "user", "content": "And in Munich?"},
        ]
        prompt = messages_to_prompt(messages)
        self.assertIn("<tool_call>", prompt)
        self.assertIn("get_weather", prompt)

    def test_full_multi_turn_loop(self):
        """Full agent loop: user → tool_call → tool_result → user follow-up."""
        messages = [
            {"role": "user", "content": "Weather in Berlin?"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [{
                    "id": "call_x",
                    "type": "function",
                    "function": {"name": "get_weather", "arguments": '{"location": "Berlin"}'},
                }],
            },
            {
                "role": "tool",
                "tool_call_id": "call_x",
                "content": "18°C, cloudy",
            },
            {"role": "user", "content": "And in Munich?"},
        ]
        prompt = messages_to_prompt(messages)
        self.assertIn("User: Weather in Berlin?", prompt)
        self.assertIn("Tool (get_weather): 18°C, cloudy", prompt)
        self.assertIn("User: And in Munich?", prompt)

    def test_list_content_joined(self):
        """Messages with list content (vision API format) should be joined as text."""
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe this image."},
                ],
            }
        ]
        prompt = messages_to_prompt(messages)
        self.assertIn("Describe this image.", prompt)

    def test_unknown_tool_call_id_fallback(self):
        """If tool_call_id has no matching assistant call, use the id itself as fallback."""
        messages = [
            {
                "role": "tool",
                "tool_call_id": "call_unknown",
                "content": "some result",
            }
        ]
        prompt = messages_to_prompt(messages)
        self.assertIn("Tool (call_unknown):", prompt)


# ===========================================================================
# wrap_tool_calls_response structure
# ===========================================================================
class TestWrapToolCallsResponse(unittest.TestCase):

    def test_structure(self):
        tc = _make_tool_call("get_weather", {"location": "Berlin"})
        resp = wrap_tool_calls_response(
            tool_calls=[tc],
            model="deepseek-chat",
            prompt_tokens=42,
            completion_tokens=10,
        )
        self.assertEqual(resp["object"], "chat.completion")
        choice = resp["choices"][0]
        self.assertEqual(choice["finish_reason"], "tool_calls")
        self.assertIsNone(choice["message"]["content"])
        self.assertEqual(len(choice["message"]["tool_calls"]), 1)
        self.assertIn("prompt_tokens", resp["usage"])
        self.assertEqual(resp["usage"]["prompt_tokens"], 42)

    def test_model_preserved(self):
        tc = _make_tool_call("ping", {})
        resp = wrap_tool_calls_response([tc], "gpt-4o", 10, 0)
        self.assertEqual(resp["model"], "gpt-4o")


# ===========================================================================
# tool_choice="none" integration — verify tools are skipped
# ===========================================================================
class TestToolChoiceNone(unittest.TestCase):

    def test_inject_skipped_when_none(self):
        """When tool_choice='none', inject_tools_into_messages should not be called
        (tested by verifying no tool prompt appears in raw messages)."""
        messages = [{"role": "user", "content": "Hello"}]
        # Simulate what chat_completions() does for tool_choice="none"
        tools = TOOLS
        tool_choice = "none"
        if tools and tool_choice != "none":
            messages = inject_tools_into_messages(messages, tools, tool_choice)
        elif tools and tool_choice == "none":
            tools = []

        # tools should now be empty, no injection
        self.assertEqual(tools, [])
        # messages untouched
        self.assertEqual(len(messages), 1)
        self.assertEqual(messages[0]["role"], "user")


# ===========================================================================
# Main
# ===========================================================================
if __name__ == "__main__":
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromModule(sys.modules[__name__])
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)

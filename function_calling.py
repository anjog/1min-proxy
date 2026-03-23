"""
Function-calling emulation for 1min-proxy.

1min.ai does not support native OpenAI function calling.
This module emulates it transparently:
  1. Inject: serializes tool descriptions into the system prompt
  2. Parse:  detects tool call patterns in model output
  3. Wrap:   transforms detected calls into the OpenAI tool_calls response format

Supported output patterns (model-dependent):
  All models: <tool_call>{"name": "...", "arguments": {...}}</tool_call>  (instructed)
  Anthropic:  <function_calls><invoke name="..."><parameter ...>  (legacy XML)
  OpenAI:     {"tool_call": {"name": "...", "arguments": {...}}}  (JSON fallback)
  DeepSeek:   <|tool▁call▁begin|>{...}<|tool▁call▁end|>
  Mistral:    [TOOL_CALLS] [{"name": "...", "arguments": {...}}]
  Llama/Sonar: <|python_tag|>{"name": "...", "parameters": {...}}
  Qwen (old): ✿FUNCTION✿: name ✿ARGS✿: {...}
"""

import re
import json
import uuid
import logging
import time

logger = logging.getLogger("1min-proxy")

# Pre-compiled tool-call detection patterns (in priority order)
_RE_P1 = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)
_RE_P2 = re.compile(
    r'\{\s*"tool_call"\s*:\s*\{[^}]*"name"\s*:\s*"([^"]+)"[^}]*"arguments"\s*:\s*(\{.*?\})',
    re.DOTALL,
)
_RE_P3_INVOKE = re.compile(r'<invoke\s+name="([^"]+)">(.*?)</invoke>', re.DOTALL)
_RE_P3_PARAM  = re.compile(r'<parameter\s+name="([^"]+)">(.*?)</parameter>', re.DOTALL)
_RE_P4 = re.compile(
    r"<\|tool[▁\s]call[▁\s]begin\|>(.*?)<\|tool[▁\s]call[▁\s]end\|>", re.DOTALL
)
_RE_P5 = re.compile(r"\[TOOL_CALLS\]\s*(\[.*?\])", re.DOTALL)
_RE_P6 = re.compile(r"<\|python_tag\|>(.*?)(?:<\|[a-z_]+\|>|$)", re.DOTALL)
_RE_P7 = re.compile(r"✿FUNCTION✿:\s*(\S+)\s*✿ARGS✿:\s*(\{.*?\})", re.DOTALL)


def build_tool_system_prompt(tools: list, tool_choice=None) -> str:
    """
    Serializes the OpenAI tools array into a system-prompt block.
    The model is instructed to output tool calls in a specific XML+JSON format
    that is unambiguous and parseable across Anthropic, OpenAI, and DeepSeek models.

    Format chosen: <tool_call>{"name": "...", "arguments": {...}}</tool_call>
    - Anthropic models produce this natively when instructed
    - OpenAI models reliably follow explicit format instructions
    - DeepSeek models respond well to XML-delimited output instructions

    tool_choice influences the mandatory/optional call instruction:
      "auto" / None  → model decides (default)
      "required"     → model MUST call a tool
      {"type": "function", "function": {"name": "..."}} → must call that specific tool
    """
    tool_descriptions = []
    for tool in tools:
        # Tools can be bare function dicts or wrapped in {"type": "function", "function": {...}}
        fn = tool.get("function", tool)
        name = fn.get("name", "")
        description = fn.get("description", "")
        params = fn.get("parameters", {})
        tool_descriptions.append(
            f"- {name}: {description}\n  Parameters: {json.dumps(params, ensure_ascii=False)}"
        )

    tools_block = "\n".join(tool_descriptions)

    if tool_choice == "required":
        call_instruction = (
            "- You MUST call one of the available tools. "
            "Do NOT respond with plain text — always output a <tool_call> block."
        )
    elif isinstance(tool_choice, dict):
        forced_name = tool_choice.get("function", {}).get("name", "")
        if forced_name:
            call_instruction = (
                f"- You MUST call the '{forced_name}' tool. "
                "Do NOT call any other tool and do NOT respond with plain text."
            )
        else:
            call_instruction = "- If no tool call is needed, respond normally without any <tool_call> block."
    else:
        call_instruction = "- If no tool call is needed, respond normally without any <tool_call> block."

    return f"""You have access to the following tools:

{tools_block}

TOOL CALL INSTRUCTIONS (mandatory):
- When you need to call a tool, respond ONLY with a tool call block — no surrounding text.
- Use this exact format:
  <tool_call>
  {{"name": "TOOL_NAME", "arguments": {{...}}}}
  </tool_call>
- The JSON inside <tool_call> must be valid. Use double quotes. No trailing commas.
{call_instruction}"""


def inject_tools_into_messages(messages: list, tools: list, tool_choice=None) -> list:
    """
    Prepends the tool description block to the system message, or inserts
    a new system message at position 0 if none exists.
    Returns a new messages list — does not mutate the original.
    """
    tool_prompt = build_tool_system_prompt(tools, tool_choice)
    patched = []
    injected = False

    for msg in messages:
        if msg.get("role") == "system" and not injected:
            # Append to existing system message
            existing = msg.get("content", "")
            patched.append({**msg, "content": f"{existing}\n\n{tool_prompt}"})
            injected = True
        else:
            patched.append(msg)

    if not injected:
        # No system message present — prepend one
        patched.insert(0, {"role": "system", "content": tool_prompt})

    return patched


def parse_tool_calls(text: str) -> list | None:
    """
    Scans model output for tool call patterns and returns an OpenAI-compatible
    tool_calls list, or None if no tool call was detected.

    Patterns handled (in order of priority):
      1. <tool_call>{...}</tool_call>              — instructed format (all models)
      2. {"tool_call": {"name":..,"arguments":..}} — OpenAI plain JSON fallback
      3. <function_calls><invoke name="...">       — legacy Anthropic XML
      4. <|tool▁call▁begin|>...<|tool▁call▁end|>  — DeepSeek native tokens
      5. [TOOL_CALLS] [{...}]                      — Mistral native format
      6. <|python_tag|>{...}                       — Llama 3.1+ / Perplexity Sonar
      7. ✿FUNCTION✿: name ✿ARGS✿: {...}           — Qwen (older checkpoints)
    """
    tool_calls = []

    # --- Pattern 1: <tool_call>{...}</tool_call> (instructed format, all models) ---
    for match in _RE_P1.finditer(text):
        try:
            data = json.loads(match.group(1))
            name = data.get("name", "")
            arguments = data.get("arguments", {})
            if name:
                tool_calls.append(_make_tool_call(name, arguments))
                logger.debug("FC-Emulation: pattern 1 (xml+json) → %s", name)
        except json.JSONDecodeError as exc:
            logger.warning("FC-Emulation: JSON parse error in pattern 1: %s", exc)

    if tool_calls:
        return tool_calls

    # --- Pattern 2: {"tool_call": {"name": ..., "arguments": ...}} ---
    for match in _RE_P2.finditer(text):
        try:
            name = match.group(1)
            arguments = json.loads(match.group(2))
            tool_calls.append(_make_tool_call(name, arguments))
            logger.debug("FC-Emulation: pattern 2 (plain json) → %s", name)
        except json.JSONDecodeError as exc:
            logger.warning("FC-Emulation: JSON parse error in pattern 2: %s", exc)

    if tool_calls:
        return tool_calls

    # --- Pattern 3: <function_calls><invoke name="..."> (legacy Anthropic XML) ---
    for match in _RE_P3_INVOKE.finditer(text):
        try:
            name = match.group(1)
            # Parameters are encoded as <parameter name="key">value</parameter>
            params_raw = match.group(2)
            arguments = {}
            for param in _RE_P3_PARAM.finditer(params_raw):
                key = param.group(1)
                value = param.group(2).strip()
                # Try to parse as JSON, fall back to string
                try:
                    arguments[key] = json.loads(value)
                except json.JSONDecodeError:
                    arguments[key] = value
            tool_calls.append(_make_tool_call(name, arguments))
            logger.debug("FC-Emulation: pattern 3 (legacy xml) → %s", name)
        except Exception as exc:
            logger.warning("FC-Emulation: parse error in pattern 3: %s", exc)

    if tool_calls:
        return tool_calls

    # --- Pattern 4: DeepSeek <|tool▁call▁begin|> format ---
    # DeepSeek R1/V3 uses Unicode "word joiner" chars in its tool tags
    for match in _RE_P4.finditer(text):
        try:
            data = json.loads(match.group(1).strip())
            name = data.get("name", "")
            arguments = data.get("arguments", data.get("parameters", {}))
            if name:
                tool_calls.append(_make_tool_call(name, arguments))
                logger.debug("FC-Emulation: pattern 4 (deepseek) → %s", name)
        except json.JSONDecodeError as exc:
            logger.warning("FC-Emulation: JSON parse error in pattern 4: %s", exc)

    if tool_calls:
        return tool_calls

    # --- Pattern 5: Mistral [TOOL_CALLS] [{...}] ---
    # Mistral models output a JSON array after the [TOOL_CALLS] sentinel.
    # Each element has "name" and "arguments".
    for match in _RE_P5.finditer(text):
        try:
            calls = json.loads(match.group(1))
            if not isinstance(calls, list):
                calls = [calls]
            for call in calls:
                name = call.get("name", "")
                arguments = call.get("arguments", call.get("parameters", {}))
                if name:
                    tool_calls.append(_make_tool_call(name, arguments))
                    logger.debug("FC-Emulation: pattern 5 (mistral) → %s", name)
        except json.JSONDecodeError as exc:
            logger.warning("FC-Emulation: JSON parse error in pattern 5: %s", exc)

    if tool_calls:
        return tool_calls

    # --- Pattern 6: Llama 3.1+ / Perplexity Sonar <|python_tag|>{...} ---
    # Llama 3.1 uses <|python_tag|> as a prefix for tool/code calls.
    # The JSON payload uses "parameters" instead of "arguments".
    # End is marked by <|eom_id|> or another special token, or end of string.
    for match in _RE_P6.finditer(text):
        try:
            data = json.loads(match.group(1).strip())
            name = data.get("name", "")
            arguments = data.get("parameters", data.get("arguments", {}))
            if name:
                tool_calls.append(_make_tool_call(name, arguments))
                logger.debug("FC-Emulation: pattern 6 (llama) → %s", name)
        except json.JSONDecodeError as exc:
            logger.warning("FC-Emulation: JSON parse error in pattern 6: %s", exc)

    if tool_calls:
        return tool_calls

    # --- Pattern 7: Qwen (older checkpoints) ✿FUNCTION✿ / ✿ARGS✿ ---
    # Older Qwen models (pre-2.5) use a proprietary marker format.
    # Qwen 2.5+ generally follows the instructed <tool_call> format instead.
    for match in _RE_P7.finditer(text):
        try:
            name = match.group(1).strip()
            arguments = json.loads(match.group(2))
            tool_calls.append(_make_tool_call(name, arguments))
            logger.debug("FC-Emulation: pattern 7 (qwen-old) → %s", name)
        except json.JSONDecodeError as exc:
            logger.warning("FC-Emulation: JSON parse error in pattern 7: %s", exc)

    return tool_calls if tool_calls else None


def _make_tool_call(name: str, arguments: dict | str) -> dict:
    """Builds a single OpenAI-format tool_call dict."""
    # arguments must be a JSON string in the OpenAI format
    if isinstance(arguments, str):
        args_str = arguments
    else:
        args_str = json.dumps(arguments, ensure_ascii=False)
    return {
        "id": f"call_{uuid.uuid4().hex[:24]}",  # OpenAI uses 24-char hex IDs
        "type": "function",
        "function": {
            "name": name,
            "arguments": args_str,
        },
    }


def wrap_tool_calls_response(
    tool_calls: list, model: str, prompt_tokens: int, completion_tokens: int
) -> dict:
    """
    Builds a complete OpenAI chat.completion response with tool_calls.
    content is set to None per OpenAI spec when tool_calls are present.
    finish_reason is set to "tool_calls" instead of "stop".
    """
    logger.info(
        "FC-Emulation: returning %d tool call(s) for model %s", len(tool_calls), model
    )
    return {
        "id": f"chatcmpl-{uuid.uuid4()}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": None,          # None per OpenAI spec when tool_calls present
                "tool_calls": tool_calls,
            },
            "finish_reason": "tool_calls",  # Signals to caller that tools need execution
        }],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }

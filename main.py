"""
1min-proxy — OpenAI-compatible proxy for 1min.ai

Based on the new Chat-with-AI API (UNIFY_CHAT_WITH_AI, March 2026).
Not a fork — standalone reimplementation.

New API vs. legacy:
  Legacy:  POST /api/features          type=CHAT_WITH_AI           Raw text stream
  New:     POST /api/chat-with-ai      type=UNIFY_CHAT_WITH_AI     Structured SSE

Streaming events of the new API:
  event: content  → data: {"content": "..."}    Text chunk
  event: result   → data: {"aiRecord": {...}}   Final metadata
  event: done     → data: {"message": "..."}    Stream finished
  event: error    → data: {"error": "..."}      Error

Function-Calling Emulation (added 2026-03):
  1min.ai does not support native OpenAI function calling.
  When tools are present in the request, this proxy:
    1. Injects a structured tool description block into the system prompt
    2. Instructs the model to respond with a specific XML/JSON format when calling a tool
    3. Parses the model response for tool call patterns
    4. Transforms detected tool calls into the OpenAI tool_calls response format
  Supported output patterns (model-dependent):
    All models: <tool_call>{"name": "...", "arguments": {...}}</tool_call>  (instructed)
    Anthropic:  <function_calls><invoke name="..."><parameter ...>  (legacy XML)
    OpenAI:     {"tool_call": {"name": "...", "arguments": {...}}}  (JSON fallback)
    DeepSeek:   <|tool▁call▁begin|>{...}<|tool▁call▁end|>
    Mistral:    [TOOL_CALLS] [{"name": "...", "arguments": {...}}]
    Llama/Sonar: <|python_tag|>{"name": "...", "parameters": {...}}
    Qwen (old): ✿FUNCTION✿: name ✿ARGS✿: {...}
"""

from dotenv import load_dotenv
load_dotenv()

from flask import Flask, request, jsonify, make_response, Response
import requests
import time
import uuid
import json
import re
import socket
import os
import logging
import base64
from io import BytesIO
import warnings

from waitress import serve
import tiktoken
import coloredlogs
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

warnings.filterwarnings("ignore", category=UserWarning, module="flask_limiter.extension")

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logger = logging.getLogger("1min-proxy")
coloredlogs.install(level=os.getenv("LOG_LEVEL", "DEBUG"), logger=logger)


# ---------------------------------------------------------------------------
# Token counting
# ---------------------------------------------------------------------------
def calculate_token(text: str, model: str = "DEFAULT") -> int:
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")
    return len(encoding.encode(text))


# ---------------------------------------------------------------------------
# Flask + Rate Limiter
# ---------------------------------------------------------------------------
app = Flask(__name__)
app.config["JSON_AS_ASCII"] = False


def _init_limiter() -> Limiter:
    host = os.getenv("MEMCACHED_HOST", "memcached")
    port = int(os.getenv("MEMCACHED_PORT", "11211"))
    try:
        from pymemcache.client.base import Client
        c = Client((host, port))
        c.set("_probe", "1")
        if c.get("_probe") == b"1":
            c.delete("_probe")
            logger.info("Rate limiter: Memcached backend (%s:%d)", host, port)
            return Limiter(
                get_remote_address, app=app,
                storage_uri=f"memcached://{host}:{port}",
            )
    except Exception:
        pass
    logger.warning("Memcached unavailable — falling back to in-memory rate limiting")
    return Limiter(get_remote_address, app=app)


limiter = _init_limiter()


# ---------------------------------------------------------------------------
# 1min.ai endpoints (new Chat API)
# ---------------------------------------------------------------------------
ONEMIN_CHAT_URL        = "https://api.1min.ai/api/chat-with-ai"
ONEMIN_CHAT_STREAM_URL = "https://api.1min.ai/api/chat-with-ai?isStreaming=true"
ONEMIN_ASSET_URL       = "https://api.1min.ai/api/assets"


# ---------------------------------------------------------------------------
# Model lists
# ---------------------------------------------------------------------------
ALL_MODELS = [
    # --- Alibaba ---
    "qwen3-vl-plus", "qwen3-vl-flash", "qwen3-max",
    "qwen-vl-plus", "qwen-vl-max", "qwen-plus", "qwen-max", "qwen-flash",

    # --- Anthropic ---
    "claude-sonnet-4-5-20250929", "claude-sonnet-4-20250514",
    "claude-opus-4-5-20251101", "claude-opus-4-20250514", "claude-opus-4-1-20250805",
    "claude-haiku-4-5-20251001",

    # --- Cohere ---
    "command-r-08-2024",

    # --- DeepSeek ---
    "deepseek-reasoner", "deepseek-chat",

    # --- Google ---
    "gemini-3-pro-preview", "gemini-3-flash-preview",
    "gemini-2.5-pro", "gemini-2.5-flash",

    # --- Mistral ---
    "magistral-small-latest", "magistral-medium-latest",
    "ministral-14b-latest", "open-mistral-nemo",
    "mistral-small-latest", "mistral-medium-latest", "mistral-large-latest",

    # --- OpenAI ---
    "gpt-5.1-codex-mini", "gpt-5.1-codex",
    "o4-mini", "o3-mini",
    "gpt-5.2-pro", "gpt-5.2", "gpt-5.1",
    "gpt-5-nano", "gpt-5-mini", "gpt-5-chat-latest", "gpt-5",
    "gpt-4o-mini", "gpt-4o",
    "gpt-4.1-nano", "gpt-4.1-mini", "gpt-4.1",
    "gpt-4-turbo", "gpt-3.5-turbo",
    "o4-mini-deep-research", "o3-pro", "o3-deep-research", "o3",

    # --- Perplexity ---
    "sonar-reasoning-pro", "sonar-pro", "sonar-deep-research", "sonar",

    # --- xAI ---
    "grok-4-fast-reasoning", "grok-4-fast-non-reasoning", "grok-4-0709",
    "grok-3-mini", "grok-3",

    # --- Meta ---
    "meta/meta-llama-3.1-405b-instruct", "meta/meta-llama-3-70b-instruct",
    "meta/llama-4-scout-instruct", "meta/llama-4-maverick-instruct",
    "meta/llama-2-70b-chat",

    # --- OpenAI OSS ---
    "openai/gpt-oss-20b", "openai/gpt-oss-120b",
]

VISION_MODELS = {
    "gpt-4o", "gpt-4o-mini", "gpt-4-turbo",
    "gpt-4.1", "gpt-4.1-mini",
    "gpt-5", "gpt-5-mini", "gpt-5.1", "gpt-5.2",
    "gemini-2.5-pro", "gemini-2.5-flash",
    "gemini-3-pro-preview", "gemini-3-flash-preview",
    "claude-sonnet-4-5-20250929", "claude-sonnet-4-20250514",
    "claude-opus-4-5-20251101", "claude-opus-4-20250514",
    "claude-haiku-4-5-20251001",
    "qwen3-vl-plus", "qwen3-vl-flash",
    "qwen-vl-plus", "qwen-vl-max",
}

DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "deepseek-chat")

_subset = os.getenv("PERMITTED_MODELS", "")
PERMITTED_MODELS  = [m.strip() for m in _subset.split(",") if m.strip()] or ALL_MODELS
RESTRICT_MODELS   = os.getenv("RESTRICT_TO_PERMITTED", "false").lower() == "true"
AVAILABLE_MODELS  = PERMITTED_MODELS if RESTRICT_MODELS else ALL_MODELS


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def extract_api_key() -> str | None:
    auth = request.headers.get("Authorization", "")
    if auth.startswith("Bearer ") and len(auth) > 7:
        return auth[7:]
    return None


def openai_error(message: str, error_type: str, code: str | None, http_status: int):
    return jsonify({
        "error": {"message": message, "type": error_type, "param": None, "code": code}
    }), http_status


def messages_to_prompt(messages: list) -> str:
    """
    Converts the OpenAI messages array into a flat prompt string.

    Handles multi-turn tool call sequences:
      - assistant messages with tool_calls (content=None) are rendered back
        in the instructed <tool_call> XML+JSON format so the model understands
        what it previously called.
      - role="tool" messages (tool results) are rendered as "Tool (name): result",
        using a pre-built map of tool_call_id → tool_name for context.
    """
    # Pre-build map of tool_call_id → tool_name for tool result formatting
    tool_call_map: dict[str, str] = {}
    for msg in messages:
        for tc in msg.get("tool_calls") or []:
            tc_id = tc.get("id", "")
            name = tc.get("function", {}).get("name", "")
            if tc_id and name:
                tool_call_map[tc_id] = name

    parts = []
    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content") or ""

        if role == "tool":
            # Tool result — look up tool name via tool_call_id for context
            tc_id = msg.get("tool_call_id", "")
            tool_name = tool_call_map.get(tc_id, tc_id or "unknown")
            if isinstance(content, list):
                content = "\n".join(
                    item.get("text", "")
                    for item in content
                    if isinstance(item, dict) and "text" in item
                )
            parts.append(f"Tool ({tool_name}): {content}")
            continue

        if role == "assistant":
            tool_calls = msg.get("tool_calls")
            if tool_calls and not content:
                # Reproduce in the instructed XML+JSON format so the model
                # recognises its own previous output when reading history
                call_strs = []
                for tc in tool_calls:
                    fn = tc.get("function", {})
                    name = fn.get("name", "")
                    args = fn.get("arguments", "{}")
                    call_strs.append(
                        f'<tool_call>{{"name": "{name}", "arguments": {args}}}</tool_call>'
                    )
                parts.append(f"Assistant: {''.join(call_strs)}")
                continue

        if isinstance(content, list):
            content = "\n".join(
                item.get("text", "")
                for item in content
                if isinstance(item, dict) and "text" in item
            )
        parts.append(f"{role.capitalize()}: {content}")

    if len(messages) > 1:
        parts.append(
            "\nRespond normally. Do NOT prefix output with 'User:' or 'Assistant:'."
        )
    return "\n".join(parts)


def upload_image(url_or_b64: str, api_key: str) -> str | None:
    """Uploads an image to the 1min.ai Asset API and returns the asset path."""
    try:
        if url_or_b64.startswith("data:image/"):
            binary_io = BytesIO(base64.b64decode(url_or_b64.split(",", 1)[1]))
        else:
            r = requests.get(url_or_b64, timeout=30)
            r.raise_for_status()
            binary_io = BytesIO(r.content)

        files = {"asset": (f"proxy-{uuid.uuid4()}.png", binary_io, "image/png")}
        resp = requests.post(
            ONEMIN_ASSET_URL, files=files,
            headers={"API-KEY": api_key}, timeout=60,
        )
        resp.raise_for_status()
        return resp.json()["fileContent"]["path"]
    except Exception as exc:
        logger.warning("Image upload failed: %s", str(exc)[:120])
        return None


def build_payload(model: str, prompt: str, image_paths: list[str]) -> dict:
    """Builds the request body for UNIFY_CHAT_WITH_AI."""
    prompt_obj: dict = {
        "prompt": prompt,
        "settings": {
            "historySettings": {
                "isMixed": False,
                "historyMessageLimit": 10,
            },
        },
    }
    if image_paths:
        prompt_obj["attachments"] = {"images": image_paths}

    return {
        "type": "UNIFY_CHAT_WITH_AI",
        "model": model,
        "promptObject": prompt_obj,
    }


# ===========================================================================
# Function-Calling Emulation
# ===========================================================================

def build_tool_system_prompt(tools: list) -> str:
    """
    Serializes the OpenAI tools array into a system-prompt block.
    The model is instructed to output tool calls in a specific XML+JSON format
    that is unambiguous and parseable across Anthropic, OpenAI, and DeepSeek models.

    Format chosen: <tool_call>{"name": "...", "arguments": {...}}</tool_call>
    - Anthropic models produce this natively when instructed
    - OpenAI models reliably follow explicit format instructions
    - DeepSeek models respond well to XML-delimited output instructions
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

    return f"""You have access to the following tools:

{tools_block}

TOOL CALL INSTRUCTIONS (mandatory):
- When you need to call a tool, respond ONLY with a tool call block — no surrounding text.
- Use this exact format:
  <tool_call>
  {{"name": "TOOL_NAME", "arguments": {{...}}}}
  </tool_call>
- The JSON inside <tool_call> must be valid. Use double quotes. No trailing commas.
- If no tool call is needed, respond normally without any <tool_call> block."""


def inject_tools_into_messages(messages: list, tools: list) -> list:
    """
    Prepends the tool description block to the system message, or inserts
    a new system message at position 0 if none exists.
    Returns a new messages list — does not mutate the original.
    """
    tool_prompt = build_tool_system_prompt(tools)
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
    for match in re.finditer(
        r"<tool_call>\s*(\{.*?\})\s*</tool_call>",
        text,
        re.DOTALL,
    ):
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
    for match in re.finditer(
        r'\{\s*"tool_call"\s*:\s*\{[^}]*"name"\s*:\s*"([^"]+)"[^}]*"arguments"\s*:\s*(\{.*?\})',
        text,
        re.DOTALL,
    ):
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
    for match in re.finditer(
        r'<invoke\s+name="([^"]+)">(.*?)</invoke>',
        text,
        re.DOTALL,
    ):
        try:
            name = match.group(1)
            # Parameters are encoded as <parameter name="key">value</parameter>
            params_raw = match.group(2)
            arguments = {}
            for param in re.finditer(
                r'<parameter\s+name="([^"]+)">(.*?)</parameter>',
                params_raw,
                re.DOTALL,
            ):
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
    for match in re.finditer(
        r"<\|tool[▁\s]call[▁\s]begin\|>(.*?)<\|tool[▁\s]call[▁\s]end\|>",
        text,
        re.DOTALL,
    ):
        try:
            data = json.loads(match.group(1).strip())
            name = data.get("name", "")
            arguments = data.get("arguments", data.get("parameters", {}))
            if name:
                tool_calls.append(_make_tool_call(name, arguments))
                logger.debug("FC-Emulation: pattern 4 (deepseek) → %s", name)
        except json.JSONDecodeError as exc:
            logger.warning("FC-Emulation: JSON parse error in pattern 4: %s", exc)

    # --- Pattern 5: Mistral [TOOL_CALLS] [{...}] ---
    # Mistral models output a JSON array after the [TOOL_CALLS] sentinel.
    # Each element has "name" and "arguments".
    for match in re.finditer(r"\[TOOL_CALLS\]\s*(\[.*?\])", text, re.DOTALL):
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
    for match in re.finditer(
        r"<\|python_tag\|>(.*?)(?:<\|[a-z_]+\|>|$)",
        text,
        re.DOTALL,
    ):
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
    for match in re.finditer(
        r"✿FUNCTION✿:\s*(\S+)\s*✿ARGS✿:\s*(\{.*?\})",
        text,
        re.DOTALL,
    ):
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


def wrap_tool_calls_response(text: str, tool_calls: list, model: str, prompt_tokens: int) -> dict:
    """
    Builds a complete OpenAI chat.completion response with tool_calls.
    content is set to None per OpenAI spec when tool_calls are present.
    finish_reason is set to "tool_calls" instead of "stop".
    """
    completion_tokens = calculate_token(text)
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


# ---------------------------------------------------------------------------
# Response-Transformer
# ---------------------------------------------------------------------------
def transform_nonstream(
    api_resp: dict,
    model: str,
    prompt_tokens: int,
    tools: list | None = None,
) -> dict:
    result = api_resp["aiRecord"]["aiRecordDetail"]["resultObject"][0]
    completion_tokens = calculate_token(result)
    logger.debug(
        "Non-streaming complete — %dp + %dc = %d tokens",
        prompt_tokens, completion_tokens, prompt_tokens + completion_tokens,
    )

    # --- Function-Calling Emulation: check response for tool calls ---
    if tools:
        detected = parse_tool_calls(result)
        if detected:
            return wrap_tool_calls_response(result, detected, model, prompt_tokens)

    return {
        "id": f"chatcmpl-{uuid.uuid4()}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": result},
            "finish_reason": "stop",
        }],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }


def stream_response(http_resp, model: str, prompt_tokens: int, tools: list | None = None):
    """
    Parses the structured SSE events from the new 1min.ai Chat API and forwards
    them in OpenAI-compatible streaming format.

    When tools are present, the full response is buffered first, then parsed
    for tool calls. If a tool call is detected, a single non-streaming-style
    chunk with tool_calls is emitted instead of content chunks.

    Note: OpenAI's streaming spec supports tool_calls in delta chunks, but
    most clients (including OpenClaw) handle the simpler single-chunk approach.
    """
    all_text = ""
    current_event: str | None = None
    completion_id = f"chatcmpl-{uuid.uuid4()}"
    created = int(time.time())

    for raw_line_bytes in http_resp.iter_lines():
        raw_line = raw_line_bytes.decode("utf-8") if isinstance(raw_line_bytes, bytes) else raw_line_bytes
        if raw_line.startswith("event:"):
            current_event = raw_line[6:].strip()

        elif raw_line.startswith("data:"):
            data_str = raw_line[5:].strip()

            if current_event == "content":
                try:
                    chunk = json.loads(data_str).get("content", "")
                except (json.JSONDecodeError, AttributeError):
                    chunk = data_str

                if chunk:
                    all_text += chunk

                    # Only stream chunks to client when NOT in tool-emulation mode
                    # (tool mode buffers the full response for parsing)
                    if not tools:
                        yield f"data: {json.dumps({
                            'id': completion_id,
                            'object': 'chat.completion.chunk',
                            'created': created,
                            'model': model,
                            'choices': [{'index': 0, 'delta': {'content': chunk},
                                         'finish_reason': None}],
                        }, ensure_ascii=False)}\n\n"

            elif current_event == "done":
                break

            elif current_event == "error":
                logger.error("1min.ai stream error: %s", data_str)
                break

            current_event = None  # Reset after each data: block

    completion_tokens = calculate_token(all_text)
    logger.debug(
        "Streaming complete — %dp + %dc = %d tokens",
        prompt_tokens, completion_tokens, prompt_tokens + completion_tokens,
    )

    # --- Function-Calling Emulation: parse buffered response ---
    if tools:
        detected = parse_tool_calls(all_text)
        if detected:
            # Emit a single chunk with tool_calls (finish_reason: tool_calls)
            tool_chunk = {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model,
                "choices": [{
                    "index": 0,
                    "delta": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": detected,
                    },
                    "finish_reason": "tool_calls",
                }],
                "usage": {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": prompt_tokens + completion_tokens,
                },
            }
            yield f"data: {json.dumps(tool_chunk, ensure_ascii=False)}\n\n"
            yield "data: [DONE]\n\n"
            return

        # No tool call detected — emit buffered text as single chunk
        yield f"data: {json.dumps({
            'id': completion_id,
            'object': 'chat.completion.chunk',
            'created': created,
            'model': model,
            'choices': [{'index': 0, 'delta': {'content': all_text},
                         'finish_reason': None}],
        }, ensure_ascii=False)}\n\n"

    # Final chunk with finish_reason=stop and usage
    yield f"data: {json.dumps({
        'id': completion_id,
        'object': 'chat.completion.chunk',
        'created': created,
        'model': model,
        'choices': [{'index': 0, 'delta': {}, 'finish_reason': 'stop'}],
        'usage': {
            'prompt_tokens': prompt_tokens,
            'completion_tokens': completion_tokens,
            'total_tokens': prompt_tokens + completion_tokens,
        },
    }, ensure_ascii=False)}\n\n"
    yield "data: [DONE]\n\n"


# ===========================================================================
# Routes
# ===========================================================================
@app.route("/")
def index():
    ip = socket.gethostbyname(socket.gethostname())
    port = os.getenv("PROXY_PORT", "5001")
    return (
        f"1min-proxy\n"
        f"  Endpoint: {ip}:{port}/v1\n"
        f"  Models:   {ip}:{port}/v1/models\n"
    )


@app.route("/v1/models")
@limiter.limit("500 per minute")
def list_models():
    data = [
        {"id": m, "object": "model", "owned_by": "1minai", "created": 1727389042}
        for m in AVAILABLE_MODELS
    ]
    return jsonify({"data": data, "object": "list"})


@app.route("/v1/chat/completions", methods=["POST", "OPTIONS"])
@limiter.limit("500 per minute")
def chat_completions():
    if request.method == "OPTIONS":
        resp = make_response()
        resp.headers.update({
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Headers": "Content-Type,Authorization",
            "Access-Control-Allow-Methods": "POST, OPTIONS",
        })
        return resp, 204

    api_key = extract_api_key()
    if not api_key:
        return openai_error("Missing API key.", "authentication_error", None, 401)

    body = request.json or {}
    model    = body.get("model", DEFAULT_MODEL)
    messages = body.get("messages", [])
    tools    = body.get("tools") or []          # OpenAI tools array (may be absent)

    if not messages:
        return openai_error(
            "No messages provided.", "invalid_request_error", "invalid_request_error", 400
        )
    if RESTRICT_MODELS and model not in AVAILABLE_MODELS:
        return openai_error(
            f"Model '{model}' not available.", "invalid_request_error", "model_not_found", 400
        )

    last_content = messages[-1].get("content")
    if not last_content:
        return openai_error(
            "Last message has no content.", "invalid_request_error", "invalid_request_error", 400
        )

    # --- Function-Calling Emulation: inject tool descriptions into messages ---
    if tools:
        logger.debug("FC-Emulation: %d tool(s) detected, injecting into system prompt", len(tools))
        messages = inject_tools_into_messages(messages, tools)

    # --- Extract images from the last message block ---
    image_paths: list[str] = []
    if isinstance(last_content, list):
        for item in last_content:
            if isinstance(item, dict) and "image_url" in item:
                if model not in VISION_MODELS:
                    return openai_error(
                        f"Model '{model}' does not support image inputs.",
                        "invalid_request_error", "model_not_supported", 400,
                    )
                path = upload_image(item["image_url"]["url"], api_key)
                if path:
                    image_paths.append(path)

    prompt        = messages_to_prompt(messages)
    prompt_tokens = calculate_token(prompt, model)
    payload       = build_payload(model, prompt, image_paths)
    logger.debug("PROMPT_TAIL: %s", prompt[-2000:])
    headers       = {"API-KEY": api_key, "Content-Type": "application/json; charset=utf-8"}

    logger.debug(
        "→ %s | %d prompt-tokens | stream=%s | tools=%d",
        model, prompt_tokens, body.get("stream"), len(tools),
    )

    if not body.get("stream", False):
        # Non-Streaming
        try:
            resp = requests.post(
                ONEMIN_CHAT_URL, json=payload, headers=headers, timeout=120
            )
            if resp.status_code == 401:
                masked = f"{api_key[:4]}...{api_key[-4:]}" if len(api_key) > 8 else "****"
                return openai_error(
                    f"Invalid API key: {masked}.", "authentication_error", "invalid_api_key", 401
                )
            resp.raise_for_status()
        except requests.exceptions.HTTPError as exc:
            logger.error("1min.ai HTTP error: %s", exc)
            return jsonify({"error": {"message": str(exc)}}), 502

        result     = transform_nonstream(resp.json(), model, prompt_tokens, tools or None)
        flask_resp = make_response(jsonify(result))
        flask_resp.headers["Content-Type"] = "application/json; charset=utf-8"
        flask_resp.headers["Access-Control-Allow-Origin"] = "*"
        return flask_resp, 200

    else:
        # Streaming
        try:
            stream_resp = requests.post(
                ONEMIN_CHAT_STREAM_URL, json=payload, headers=headers,
                stream=True, timeout=120,
            )
            if stream_resp.status_code == 401:
                masked = f"{api_key[:4]}...{api_key[-4:]}" if len(api_key) > 8 else "****"
                return openai_error(
                    f"Invalid API key: {masked}.", "authentication_error", "invalid_api_key", 401
                )
            stream_resp.raise_for_status()
        except requests.exceptions.RequestException as exc:
            logger.error("Streaming request failed: %s", exc)
            return jsonify({"error": {"message": str(exc)}}), 502

        return Response(
            stream_response(stream_resp, model, prompt_tokens, tools or None),
            content_type="text/event-stream",
        )


# ===========================================================================
# Start
# ===========================================================================
if __name__ == "__main__":
    HOST    = os.getenv("PROXY_HOST", "0.0.0.0")
    PORT    = int(os.getenv("PROXY_PORT", "5001"))
    THREADS = int(os.getenv("PROXY_THREADS", "6"))

    ip = socket.gethostbyname(socket.gethostname())
    logger.info(
        "\n1min-proxy ready\n  http://%s:%d/v1\n  http://%s:%d/v1/models\n",
        ip, PORT, ip, PORT,
    )
    serve(app, host=HOST, port=PORT, threads=THREADS)

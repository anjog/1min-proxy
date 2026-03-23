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

Function-calling emulation (added 2026-03) is handled by function_calling.py.
Model registry and fallback model lists live in model_registry.py.
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
import threading
import logging
import base64
from io import BytesIO
import warnings

from waitress import serve
import tiktoken
import coloredlogs
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

from model_registry import ModelRegistry, FALLBACK_MODELS, FALLBACK_VISION_MODELS
from function_calling import inject_tools_into_messages, parse_tool_calls, wrap_tool_calls_response

warnings.filterwarnings("ignore", category=UserWarning, module="flask_limiter.extension")

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logger = logging.getLogger("1min-proxy")
logger.propagate = False
coloredlogs.install(
    level=os.getenv("LOG_LEVEL", "INFO"),
    logger=logger,
    fmt="%(asctime)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


# ---------------------------------------------------------------------------
# Crawling-line filter
# ---------------------------------------------------------------------------
_CRAWL_RE = re.compile(r"^🌐 Crawling site https?://[^\n]*\n?", re.MULTILINE)


def strip_crawl_lines(text: str) -> str:
    """Remove 1min.ai crawling status lines (e.g. '🌐 Crawling site <url>')."""
    cleaned = _CRAWL_RE.sub("", text)
    if cleaned != text:
        logger.debug("Filtered 1min.ai crawl line(s) from response")
    return cleaned


# ---------------------------------------------------------------------------
# Token counting
# ---------------------------------------------------------------------------
def calculate_token(text: str, model: str = "DEFAULT") -> int:
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")
    return len(encoding.encode(text))


def _log_credits(model: str, prompt_tokens: int, completion_tokens: int):
    meta = _model_registry.get_model_meta(model)
    if not meta:
        return
    cost = prompt_tokens * meta.get("INPUT", 0) / 1000 \
         + completion_tokens * meta.get("OUTPUT", 0) / 1000
    logger.info(
        "Credits: %s | %d in + %d out = %d tokens | ~%.4f Credits",
        model, prompt_tokens, completion_tokens,
        prompt_tokens + completion_tokens, cost,
    )


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
# Model registry config
# ---------------------------------------------------------------------------
_subset = os.getenv("PERMITTED_MODELS", "")
_PERMITTED_MODELS = [m.strip() for m in _subset.split(",") if m.strip()]
RESTRICT_MODELS   = os.getenv("RESTRICT_TO_PERMITTED", "false").lower() == "true"

_MODELS_BASE_URL = os.getenv("ONEMIN_MODELS_URL", "https://api.1min.ai/models")
_MODEL_CACHE_TTL = int(os.getenv("MODEL_CACHE_TTL", "300"))


_model_registry = ModelRegistry(
    base_url  = _MODELS_BASE_URL,
    ttl       = _MODEL_CACHE_TTL,
    permitted = _PERMITTED_MODELS,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def extract_api_key() -> str | None:
    auth = request.headers.get("Authorization", "")
    if auth.startswith("Bearer ") and len(auth) > 7:
        return auth[7:]
    return None


def _mask_api_key(key: str) -> str:
    return f"{key[:4]}...{key[-4:]}" if len(key) > 8 else "****"


def _flatten_content(content: list) -> str:
    return "\n".join(
        item.get("text", "")
        for item in content
        if isinstance(item, dict) and "text" in item
    )


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
                content = _flatten_content(content)
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
            content = _flatten_content(content)
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


# ---------------------------------------------------------------------------
# Response-Transformer
# ---------------------------------------------------------------------------
def transform_nonstream(
    api_resp: dict,
    model: str,
    prompt_tokens: int,
    tools: list | None = None,
) -> dict:
    try:
        result = api_resp["aiRecord"]["aiRecordDetail"]["resultObject"][0]
    except (KeyError, IndexError, TypeError) as exc:
        logger.error("Unexpected 1min.ai response structure: %s", exc)
        return {
            "error": {
                "message": "Unexpected upstream response structure.",
                "type": "server_error",
                "param": None,
                "code": "upstream_error",
            }
        }
    result = strip_crawl_lines(result)
    completion_tokens = calculate_token(result)
    logger.debug(
        "Non-streaming complete — %dp + %dc = %d tokens",
        prompt_tokens, completion_tokens, prompt_tokens + completion_tokens,
    )
    _log_credits(model, prompt_tokens, completion_tokens)

    # --- Function-Calling Emulation: check response for tool calls ---
    if tools:
        detected = parse_tool_calls(result)
        if detected:
            return wrap_tool_calls_response(detected, model, prompt_tokens, completion_tokens)

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

                chunk = strip_crawl_lines(chunk)

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
    _log_credits(model, prompt_tokens, completion_tokens)

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
    available = _model_registry.get_available_models()
    data = [
        {"id": m, "object": "model", "owned_by": "1minai", "created": 1727389042}
        for m in available
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
    model       = body.get("model")
    messages    = body.get("messages", [])
    tools       = body.get("tools") or []          # OpenAI tools array (may be absent)
    tool_choice = body.get("tool_choice", "auto")  # "auto" | "none" | "required" | {..}

    if not model:
        return openai_error(
            "No model specified.", "invalid_request_error", "invalid_request_error", 400
        )
    if not messages:
        return openai_error(
            "No messages provided.", "invalid_request_error", "invalid_request_error", 400
        )
    if RESTRICT_MODELS and model not in _model_registry.get_available_models():
        return openai_error(
            f"Model '{model}' not available.", "invalid_request_error", "model_not_found", 400
        )

    _meta      = _model_registry.get_model_meta(model)
    max_tokens = body.get("max_tokens")
    if _meta and max_tokens and max_tokens > _meta.get("MAX_OUTPUT_TOKEN", float("inf")):
        return openai_error(
            f"max_tokens {max_tokens} exceeds model limit of {_meta['MAX_OUTPUT_TOKEN']}.",
            "invalid_request_error", "invalid_request_error", 400,
        )

    last_content = messages[-1].get("content")
    if not last_content:
        return openai_error(
            "Last message has no content.", "invalid_request_error", "invalid_request_error", 400
        )

    # --- Function-Calling Emulation: inject tool descriptions into messages ---
    if tools and tool_choice != "none":
        logger.debug(
            "FC-Emulation: %d tool(s) detected, tool_choice=%r, injecting into system prompt",
            len(tools), tool_choice,
        )
        messages = inject_tools_into_messages(messages, tools, tool_choice)
    elif tools and tool_choice == "none":
        logger.debug("FC-Emulation: tool_choice=none — skipping tool injection")
        tools = []  # Treat as tool-free request for response parsing too

    # --- Extract images from the last message block ---
    image_paths: list[str] = []
    if isinstance(last_content, list):
        for item in last_content:
            if isinstance(item, dict) and "image_url" in item:
                if not _model_registry.is_vision_model(model):
                    return openai_error(
                        f"Model '{model}' does not support image inputs.",
                        "invalid_request_error", "model_not_supported", 400,
                    )
                path = upload_image(item["image_url"]["url"], api_key)
                if path:
                    image_paths.append(path)

    prompt        = messages_to_prompt(messages)
    prompt_tokens = calculate_token(prompt, model)
    if _meta and prompt_tokens > _meta.get("CONTEXT", float("inf")):
        logger.warning(
            "Prompt (%d tokens) überschreitet Kontextfenster von %s (%d tokens)",
            prompt_tokens, model, _meta["CONTEXT"],
        )
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
                return openai_error(
                    f"Invalid API key: {_mask_api_key(api_key)}.", "authentication_error", "invalid_api_key", 401
                )
            resp.raise_for_status()
        except requests.exceptions.HTTPError as exc:
            logger.error("1min.ai HTTP error: %s", exc)
            return jsonify({"error": {"message": str(exc)}}), 502

        result     = transform_nonstream(resp.json(), model, prompt_tokens, tools)
        http_status = 502 if "error" in result else 200
        flask_resp = make_response(jsonify(result))
        flask_resp.headers["Content-Type"] = "application/json; charset=utf-8"
        flask_resp.headers["Access-Control-Allow-Origin"] = "*"
        return flask_resp, http_status

    else:
        # Streaming
        try:
            stream_resp = requests.post(
                ONEMIN_CHAT_STREAM_URL, json=payload, headers=headers,
                stream=True, timeout=120,
            )
            if stream_resp.status_code == 401:
                return openai_error(
                    f"Invalid API key: {_mask_api_key(api_key)}.", "authentication_error", "invalid_api_key", 401
                )
            stream_resp.raise_for_status()
        except requests.exceptions.RequestException as exc:
            logger.error("Streaming request failed: %s", exc)
            return jsonify({"error": {"message": str(exc)}}), 502

        return Response(
            stream_response(stream_resp, model, prompt_tokens, tools),
            content_type="text/event-stream",
        )


# ===========================================================================
# Start
# ===========================================================================
if __name__ == "__main__":
    HOST    = os.getenv("PROXY_HOST", "0.0.0.0")
    PORT    = int(os.getenv("PROXY_PORT", "5001"))
    THREADS = int(os.getenv("PROXY_THREADS", "6"))

    # Eager Fetch beim Start (Fehler nicht fatal)
    try:
        _model_registry.refresh()
    except Exception:
        pass

    # Background-Refresh-Daemon
    def _bg_refresh_loop():
        interval = max(_MODEL_CACHE_TTL, 60)
        while True:
            time.sleep(interval)
            try:
                _model_registry.refresh()
            except Exception:
                pass

    _bg = threading.Thread(target=_bg_refresh_loop, daemon=True, name="model-registry-refresh")
    _bg.start()
    logger.info("ModelRegistry: Background-Thread gestartet (Intervall=%ds)", _MODEL_CACHE_TTL)

    ip = socket.gethostbyname(socket.gethostname())
    logger.info(
        "\n1min-proxy ready\n  http://%s:%d/v1\n  http://%s:%d/v1/models\n",
        ip, PORT, ip, PORT,
    )
    serve(app, host=HOST, port=PORT, threads=THREADS)

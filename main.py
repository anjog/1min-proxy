"""
1min-proxy — OpenAI-kompatibler Proxy für 1min.ai

Basiert auf der neuen Chat-with-AI API (UNIFY_CHAT_WITH_AI, März 2026).
Kein Fork — eigenständige Neuentwicklung.

Neue API vs. Legacy:
  Legacy:  POST /api/features          type=CHAT_WITH_AI    Roher Textstream
  Neu:     POST /api/chat-with-ai      type=UNIFY_CHAT_WITH_AI  Strukturiertes SSE

Streaming-Events der neuen API:
  event: content  → data: {"content": "..."}    Text-Chunk
  event: result   → data: {"aiRecord": {...}}   Abschluss-Metadaten
  event: done     → data: {"message": "..."}    Stream beendet
  event: error    → data: {"error": "..."}       Fehler
"""

from dotenv import load_dotenv
load_dotenv()

from flask import Flask, request, jsonify, make_response, Response
import requests
import time
import uuid
import json
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
# Token-Zählung
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


def _init_limiter() -> Limiter:
    host = os.getenv("MEMCACHED_HOST", "memcached")
    port = int(os.getenv("MEMCACHED_PORT", "11211"))
    try:
        from pymemcache.client.base import Client
        c = Client((host, port))
        c.set("_probe", "1")
        if c.get("_probe") == b"1":
            c.delete("_probe")
            logger.info("Rate Limiter: Memcached-Backend (%s:%d)", host, port)
            return Limiter(
                get_remote_address, app=app,
                storage_uri=f"memcached://{host}:{port}",
            )
    except Exception:
        pass
    logger.warning("Memcached nicht erreichbar — In-Memory Rate Limiting (nicht für Produktion)")
    return Limiter(get_remote_address, app=app)


limiter = _init_limiter()


# ---------------------------------------------------------------------------
# 1min.ai Endpunkte (neue Chat-API)
# ---------------------------------------------------------------------------
ONEMIN_CHAT_URL        = "https://api.1min.ai/api/chat-with-ai"
ONEMIN_CHAT_STREAM_URL = "https://api.1min.ai/api/chat-with-ai?isStreaming=true"
ONEMIN_ASSET_URL       = "https://api.1min.ai/api/assets"


# ---------------------------------------------------------------------------
# Modell-Listen
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
# Hilfsfunktionen
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
    """Wandelt das OpenAI-Messages-Array in einen flachen Prompt-String um."""
    parts = []
    for msg in messages:
        role = msg.get("role", "").capitalize()
        content = msg.get("content", "")
        if isinstance(content, list):
            content = "\n".join(
                item.get("text", "")
                for item in content
                if isinstance(item, dict) and "text" in item
            )
        parts.append(f"{role}: {content}")
    if len(messages) > 1:
        parts.append(
            "\nRespond normally. Do NOT prefix output with 'User:' or 'Assistant:'."
        )
    return "\n".join(parts)


def upload_image(url_or_b64: str, api_key: str) -> str | None:
    """Lädt ein Bild zur 1min.ai Asset-API hoch und gibt den Asset-Pfad zurück."""
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
        logger.warning("Bild-Upload fehlgeschlagen: %s", str(exc)[:120])
        return None


def build_payload(model: str, prompt: str, image_paths: list[str]) -> dict:
    """Erstellt den Request-Body für UNIFY_CHAT_WITH_AI."""
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
def transform_nonstream(api_resp: dict, model: str, prompt_tokens: int) -> dict:
    result = api_resp["aiRecord"]["aiRecordDetail"]["resultObject"][0]
    completion_tokens = calculate_token(result)
    logger.debug(
        "Non-streaming abgeschlossen — %dp + %dc = %d tokens",
        prompt_tokens, completion_tokens, prompt_tokens + completion_tokens,
    )
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


def stream_response(http_resp, model: str, prompt_tokens: int):
    """
    Parst die strukturierten SSE-Events der neuen 1min.ai Chat-API und gibt
    sie im OpenAI-kompatiblen Streaming-Format weiter.

    Die neue API sendet benannte Events (event: content / result / done / error).
    Das ist fundamental anders als der alte Rohtextstream der Feature-API.
    iter_lines() überspringt Leerzeilen zwischen Events automatisch.
    """
    all_text = ""
    current_event: str | None = None
    completion_id = f"chatcmpl-{uuid.uuid4()}"
    created = int(time.time())

    for raw_line in http_resp.iter_lines(decode_unicode=True):
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
                    yield f"data: {json.dumps({
                        'id': completion_id,
                        'object': 'chat.completion.chunk',
                        'created': created,
                        'model': model,
                        'choices': [{'index': 0, 'delta': {'content': chunk},
                                     'finish_reason': None}],
                    })}\n\n"

            elif current_event == "done":
                break

            elif current_event == "error":
                logger.error("1min.ai Stream-Fehler: %s", data_str)
                break

            # "result"-Event enthält nur Metadaten — ignorieren

            current_event = None  # Reset nach jedem data:-Block

    completion_tokens = calculate_token(all_text)
    logger.debug(
        "Streaming abgeschlossen — %dp + %dc = %d tokens",
        prompt_tokens, completion_tokens, prompt_tokens + completion_tokens,
    )

    # Abschluss-Chunk mit finish_reason und usage
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
    })}\n\n"
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
        f"  Modelle:  {ip}:{port}/v1/models\n"
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

    # --- Bilder aus dem letzten Message-Block extrahieren ---
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

    prompt       = messages_to_prompt(messages)
    prompt_tokens = calculate_token(prompt, model)
    payload      = build_payload(model, prompt, image_paths)
    headers      = {"API-KEY": api_key, "Content-Type": "application/json"}

    logger.debug("→ %s | %d prompt-tokens | stream=%s", model, prompt_tokens, body.get("stream"))

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
            logger.error("1min.ai HTTP-Fehler: %s", exc)
            return jsonify({"error": {"message": str(exc)}}), 502

        result    = transform_nonstream(resp.json(), model, prompt_tokens)
        flask_resp = make_response(jsonify(result))
        flask_resp.headers["Content-Type"] = "application/json"
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
            logger.error("Stream-Anfrage fehlgeschlagen: %s", exc)
            return jsonify({"error": {"message": str(exc)}}), 502

        return Response(
            stream_response(stream_resp, model, prompt_tokens),
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
        "\n1min-proxy bereit\n  http://%s:%d/v1\n  http://%s:%d/v1/models\n",
        ip, PORT, ip, PORT,
    )
    serve(app, host=HOST, port=PORT, threads=THREADS)

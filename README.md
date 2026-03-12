# 1min-proxy

🇩🇪 [Deutsche Version](README.de.md)

OpenAI-compatible proxy for [1min.ai](https://1min.ai), built on the new **Chat-with-AI API** (`UNIFY_CHAT_WITH_AI`).

A standalone reimplementation based on [1min-relay](https://github.com/kokofixcomputers/1min-relay). Two issues motivated a fresh start rather than a fork:

1. **New API:** 1min.ai replaced the old AI Feature API (`/api/features`) with a new Chat API (`/api/chat-with-ai`, type `UNIFY_CHAT_WITH_AI`). 1min-relay still uses the legacy API, which will be deprecated.
2. **Cleaner streaming:** The new API sends structured SSE events instead of a raw byte stream, making the Crawling-injection workaround from 1min-relay unnecessary (and the problem does not occur with the new API at all).

## Features

- OpenAI-compatible endpoints (`/v1/chat/completions`, `/v1/models`)
- Streaming (structured SSE events of the new API) and non-streaming
- **Function-calling emulation** — 1min.ai has no native tool support; the proxy emulates it transparently via prompt injection and response parsing, including `tool_choice` and multi-turn tool loops
- Image uploads via 1min.ai Asset API (vision models)
- **Dynamic model registry** — fetches available models from the 1min.ai API on startup and refreshes every 5 minutes; falls back to a built-in list if the API is unreachable
- **Credit cost logging** — estimates credit cost per request based on `creditMetadata` from the model API (`INPUT`/`OUTPUT` credits per 1,000 tokens)
- **Model limit validation** — enforces `MAX_OUTPUT_TOKEN` and warns when the prompt exceeds the model's context window
- Model whitelist via environment variable
- Rate limiting (Memcached or in-memory)
- Fully configurable via environment variables / `.env`

## Requirements

- Python 3.11+
- 1min.ai API key

## Installation

```bash
git clone https://github.com/anjog/1min-proxy.git
cd 1min-proxy

python3 -m venv venv
venv/bin/pip install -r requirements.txt

cp .env.example .env
# edit .env as needed
```

## Running

```bash
# Directly
venv/bin/python3 main.py
```

### systemd system service (requires sudo)

```bash
# Edit 1min-proxy.service — replace User, Group, and /path/to/1min-proxy placeholders
sudo cp 1min-proxy.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now 1min-proxy
```

### systemd user service (no sudo required)

Runs under your own user account. Starts on login; with linger enabled, also starts at boot.

```bash
# Edit 1min-proxy-user.service — replace PATH/TO/1min-proxy with your actual path
mkdir -p ~/.config/systemd/user/
cp 1min-proxy-user.service ~/.config/systemd/user/1min-proxy.service
systemctl --user daemon-reload
systemctl --user enable --now 1min-proxy

# Optional: start automatically at boot without an active login session
loginctl enable-linger $USER
```

## Configuration

| Variable | Default | Description |
|---|---|---|
| `PROXY_HOST` | `0.0.0.0` | Bind address |
| `PROXY_PORT` | `5001` | Port |
| `PROXY_THREADS` | `6` | Waitress worker threads |
| `LOG_LEVEL` | `INFO` | `DEBUG`, `INFO`, `WARNING`, `ERROR` |
| `PERMITTED_MODELS` | *(all)* | Comma-separated model whitelist (filters dynamic list) |
| `RESTRICT_TO_PERMITTED` | `false` | Reject models not on the whitelist |
| `ONEMIN_MODELS_URL` | `https://api.1min.ai/models` | 1min.ai Models API URL |
| `MODEL_CACHE_TTL` | `300` | Model list cache duration in seconds |
| `MEMCACHED_HOST` | `memcached` | Memcached host for rate limiting |
| `MEMCACHED_PORT` | `11211` | Memcached port |

## Usage

The proxy is a drop-in replacement for the OpenAI API. Use your 1min.ai API key as the Bearer token.

```bash
# List models
curl http://localhost:5001/v1/models

# Chat request (non-streaming)
curl http://localhost:5001/v1/chat/completions \
  -H "Authorization: Bearer YOUR_1MIN_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "deepseek-chat",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'

# Streaming
curl http://localhost:5001/v1/chat/completions \
  -H "Authorization: Bearer YOUR_1MIN_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "deepseek-chat",
    "stream": true,
    "messages": [{"role": "user", "content": "Hello!"}]
  }'

# Function calling (emulated — works with all models)
curl http://localhost:5001/v1/chat/completions \
  -H "Authorization: Bearer YOUR_1MIN_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "deepseek-chat",
    "messages": [{"role": "user", "content": "What is the weather in Berlin?"}],
    "tools": [{
      "type": "function",
      "function": {
        "name": "get_weather",
        "description": "Get current weather for a location",
        "parameters": {
          "type": "object",
          "properties": {"location": {"type": "string"}},
          "required": ["location"]
        }
      }
    }]
  }'
```

## New vs. Legacy API

| Aspect | Legacy (`/api/features`) | New (`/api/chat-with-ai`) |
|---|---|---|
| Type | `CHAT_WITH_AI` | `UNIFY_CHAT_WITH_AI` |
| Streaming | Raw text stream | Structured SSE events |
| Images | `promptObject.imageList` | `promptObject.attachments.images` |
| Web search | Top-level parameters | `settings.webSearchSettings` |
| History | Top-level `isMixed` | `settings.historySettings` |

## License

MIT

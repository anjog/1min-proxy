# 1min-proxy

🇩🇪 [Deutsche Version](README.de.md)

OpenAI-compatible proxy for [1min.ai](https://1min.ai), built on the new **Chat-with-AI API** (`UNIFY_CHAT_WITH_AI`).

A standalone implementation — not a fork. The goal was to migrate from the deprecated AI Feature API (`/api/features`) to the new, structured Chat API (`/api/chat-with-ai`).

## Features

- OpenAI-compatible endpoints (`/v1/chat/completions`, `/v1/models`)
- Streaming (structured SSE events of the new API) and non-streaming
- Image uploads via 1min.ai Asset API (vision models)
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

# With systemd
sudo cp 1min-proxy.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now 1min-proxy
```

## Configuration

| Variable | Default | Description |
|---|---|---|
| `PROXY_HOST` | `0.0.0.0` | Bind address |
| `PROXY_PORT` | `5001` | Port |
| `PROXY_THREADS` | `6` | Waitress worker threads |
| `DEFAULT_MODEL` | `deepseek-chat` | Model used when none is specified |
| `LOG_LEVEL` | `INFO` | `DEBUG`, `INFO`, `WARNING`, `ERROR` |
| `PERMITTED_MODELS` | *(all)* | Comma-separated model whitelist |
| `RESTRICT_TO_PERMITTED` | `false` | Reject models not on the whitelist |
| `MEMCACHED_HOST` | `memcached` | Memcached host for rate limiting |
| `MEMCACHED_PORT` | `11211` | Memcached port |

## Usage

The proxy is a drop-in replacement for the OpenAI API. Use your 1min.ai API key as the Bearer token.

```bash
# List models
curl http://localhost:5002/v1/models

# Chat request (non-streaming)
curl http://localhost:5002/v1/chat/completions \
  -H "Authorization: Bearer YOUR_1MIN_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "deepseek-chat",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'

# Streaming
curl http://localhost:5002/v1/chat/completions \
  -H "Authorization: Bearer YOUR_1MIN_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "deepseek-chat",
    "stream": true,
    "messages": [{"role": "user", "content": "Hello!"}]
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

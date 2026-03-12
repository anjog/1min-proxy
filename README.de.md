# 1min-proxy

🇬🇧 [English version](README.md)

OpenAI-kompatibler Proxy für [1min.ai](https://1min.ai), basierend auf der neuen **Chat-with-AI API** (`UNIFY_CHAT_WITH_AI`).

Eigenständige Neuentwicklung auf Basis von [1min-relay](https://github.com/kokofixcomputers/1min-relay). Zwei Probleme motivierten einen Neustart statt eines direkten Forks:

1. **Neue API:** 1min.ai hat die alte AI Feature API (`/api/features`) durch eine neue Chat-API (`/api/chat-with-ai`, Typ `UNIFY_CHAT_WITH_AI`) abgelöst. 1min-relay nutzt noch die Legacy-API, die deprecated werden wird.
2. **Saubereres Streaming:** Die neue API sendet strukturierte SSE-Events statt eines rohen Byte-Streams — der Crawling-Injektions-Workaround aus 1min-relay wird damit überflüssig (das Problem tritt mit der neuen API gar nicht mehr auf).

## Features

- OpenAI-kompatible Endpunkte (`/v1/chat/completions`, `/v1/models`)
- Streaming (strukturierte SSE-Events der neuen API) und Non-Streaming
- **Function-Calling-Emulation** — 1min.ai unterstützt kein natives Tool-Calling; der Proxy emuliert es transparent per Prompt-Injektion und Response-Parsing, inkl. `tool_choice` und Multi-Turn-Tool-Loops
- Bild-Uploads via 1min.ai Asset-API (Vision-Modelle)
- **Dynamische Modell-Registry** — lädt verfügbare Modelle beim Start von der 1min.ai API und aktualisiert sie alle 5 Minuten; fällt bei Nichterreichbarkeit auf eine eingebaute Liste zurück
- **Credit-Kosten-Logging** — schätzt die Credit-Kosten pro Request anhand der `creditMetadata` aus der Modell-API (`INPUT`/`OUTPUT` Credits pro 1.000 Token)
- **Modell-Limit-Validierung** — prüft `MAX_OUTPUT_TOKEN` und warnt wenn der Prompt das Kontextfenster des Modells überschreitet
- Modell-Whitelist per Umgebungsvariable
- Rate Limiting (Memcached oder In-Memory)
- Konfiguration vollständig über Umgebungsvariablen / `.env`

## Voraussetzungen

- Python 3.11+
- 1min.ai API-Key

## Installation

```bash
git clone https://github.com/anjog/1min-proxy.git
cd 1min-proxy

python3 -m venv venv
venv/bin/pip install -r requirements.txt

cp .env.example .env
# .env anpassen (Port, Modell etc.)
```

## Starten

```bash
# Direkt
venv/bin/python3 main.py
```

### systemd System-Service (erfordert sudo)

```bash
# 1min-proxy.service editieren — User, Group und /path/to/1min-proxy anpassen
sudo cp 1min-proxy.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now 1min-proxy
```

### systemd User-Service (kein sudo nötig)

Läuft unter dem eigenen Benutzer. Startet beim Login; mit Linger auch beim Systemstart.

```bash
# 1min-proxy-user.service editieren — PATH/TO/1min-proxy durch den echten Pfad ersetzen
mkdir -p ~/.config/systemd/user/
cp 1min-proxy-user.service ~/.config/systemd/user/1min-proxy.service
systemctl --user daemon-reload
systemctl --user enable --now 1min-proxy

# Optional: automatisch starten ohne aktive Login-Session
loginctl enable-linger $USER
```

### OpenWRT (procd)

OpenWRT nutzt procd statt systemd. Getestet auf arm64-Routern (z.B. BananaPi R4);
für arm64 sind tiktoken-Wheels verfügbar, der Proxy läuft ohne Anpassungen.

```bash
# Python installieren
opkg update
opkg install python3 python3-pip

# Proxy einrichten
mkdir -p /opt/1min-proxy
# Dateien per scp oder git clone kopieren
cd /opt/1min-proxy
python3 -m venv venv
venv/bin/pip install -r requirements.txt
cp .env.example .env
# .env anpassen

# procd-Init-Script installieren und aktivieren
cp 1min-proxy.procd /etc/init.d/1min-proxy
chmod +x /etc/init.d/1min-proxy
/etc/init.d/1min-proxy enable
/etc/init.d/1min-proxy start
```

Logs landen im System-Log und können so verfolgt werden:
```bash
logread -f -e 1min-proxy
```

## Konfiguration

| Variable | Standard | Beschreibung |
|---|---|---|
| `PROXY_HOST` | `0.0.0.0` | Bind-Adresse |
| `PROXY_PORT` | `5001` | Port |
| `PROXY_THREADS` | `6` | Waitress Worker-Threads |
| `LOG_LEVEL` | `INFO` | `DEBUG`, `INFO`, `WARNING`, `ERROR` |
| `PERMITTED_MODELS` | *(alle)* | Kommagetrennte Whitelist (filtert dynamische Liste) |
| `RESTRICT_TO_PERMITTED` | `false` | Nur Whitelist-Modelle erlauben |
| `ONEMIN_MODELS_URL` | `https://api.1min.ai/models` | 1min.ai Models-API-URL |
| `MODEL_CACHE_TTL` | `300` | Cache-Dauer der Modellliste in Sekunden |
| `MEMCACHED_HOST` | `memcached` | Memcached-Host für Rate Limiting |
| `MEMCACHED_PORT` | `11211` | Memcached-Port |

## Verwendung

Der Proxy ist als Drop-in-Ersatz für die OpenAI-API konzipiert. API-Key = dein 1min.ai API-Key.

```bash
# Modelle auflisten
curl http://localhost:5001/v1/models

# Chat-Anfrage (Non-Streaming)
curl http://localhost:5001/v1/chat/completions \
  -H "Authorization: Bearer DEIN_1MIN_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "deepseek-chat",
    "messages": [{"role": "user", "content": "Hallo!"}]
  }'

# Streaming
curl http://localhost:5001/v1/chat/completions \
  -H "Authorization: Bearer DEIN_1MIN_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "deepseek-chat",
    "stream": true,
    "messages": [{"role": "user", "content": "Hallo!"}]
  }'

# Function Calling (emuliert — funktioniert mit allen Modellen)
curl http://localhost:5001/v1/chat/completions \
  -H "Authorization: Bearer DEIN_1MIN_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "deepseek-chat",
    "messages": [{"role": "user", "content": "Wie ist das Wetter in Berlin?"}],
    "tools": [{
      "type": "function",
      "function": {
        "name": "get_weather",
        "description": "Aktuelles Wetter für einen Ort abrufen",
        "parameters": {
          "type": "object",
          "properties": {"location": {"type": "string"}},
          "required": ["location"]
        }
      }
    }]
  }'
```

## Neue vs. Legacy API

| Aspekt | Legacy (`/api/features`) | Neu (`/api/chat-with-ai`) |
|---|---|---|
| Typ | `CHAT_WITH_AI` | `UNIFY_CHAT_WITH_AI` |
| Streaming | Roher Textstream | Strukturierte SSE-Events |
| Bilder | `promptObject.imageList` | `promptObject.attachments.images` |
| Web-Suche | Top-Level-Parameter | `settings.webSearchSettings` |
| History | Top-Level `isMixed` | `settings.historySettings` |

## Lizenz

MIT

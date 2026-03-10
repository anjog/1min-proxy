#!/usr/bin/env bash
# Smoketest Step 2 — Multi-Turn Loop
# Schickt Tool-Call aus Step 1 als History zurück + simuliertes Tool-Result
# Erwartet: erneuter Tool-Call für München, oder natürliche Antwort

API_KEY="${1:-$ONEMIN_API_KEY}"

curl -s http://localhost:5001/v1/chat/completions \
  -H "Authorization: Bearer $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "deepseek-chat",
    "tools": [{
      "type": "function",
      "function": {
        "name": "get_weather",
        "description": "Get weather",
        "parameters": {
          "type": "object",
          "properties": {
            "location": {"type": "string"}
          },
          "required": ["location"]
        }
      }
    }],
    "messages": [
      {"role": "user", "content": "Wie ist das Wetter in Berlin?"},
      {"role": "assistant", "content": null, "tool_calls": [{
        "id": "call_abc",
        "type": "function",
        "function": {"name": "get_weather", "arguments": "{\"location\":\"Berlin\"}"}
      }]},
      {"role": "tool", "tool_call_id": "call_abc", "content": "{\"temperature\":18,\"condition\":\"bewölkt\"}"},
      {"role": "user", "content": "Und in München?"}
    ]
  }' | jq .

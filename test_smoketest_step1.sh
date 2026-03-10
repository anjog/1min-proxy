#!/usr/bin/env bash
# Smoketest Step 1 — Single Tool Call
# Erwartet: finish_reason="tool_calls", content=null, tool_calls[0].function.name="get_weather"

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
      {"role": "user", "content": "Wie ist das Wetter in Berlin?"}
    ]
  }' | jq .

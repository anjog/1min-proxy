# Simplify: Code Review Fixes — 2026-03-23

Changes applied during a `/simplify` review pass across `main.py`,
`function_calling.py`, and `model_registry.py`.

---

## 1. Bug: `transform_nonstream` error path returned HTTP 200

**File:** `main.py` — `chat_completions` route (non-streaming branch)

**Before:**
```python
result     = transform_nonstream(resp.json(), model, prompt_tokens, tools or None)
flask_resp = make_response(jsonify(result))
...
return flask_resp, 200
```

**After:**
```python
result      = transform_nonstream(resp.json(), model, prompt_tokens, tools)
http_status = 502 if "error" in result else 200
flask_resp  = make_response(jsonify(result))
...
return flask_resp, http_status
```

**Why:** `transform_nonstream` returns an OpenAI-format `{"error": {...}}` dict when
the upstream response has an unexpected structure. The caller was unconditionally
returning HTTP 200, so clients received a successful status code with an error body —
violating the OpenAI error contract and breaking any client that inspects the HTTP
status (rather than the body) to detect failures. Now returns 502.

---

## 2. DRY: API key masking duplicated in two 401 handlers

**File:** `main.py` — non-streaming (line ~581) and streaming (line ~604) branches

**Before:** (copy-pasted in both branches)
```python
masked = f"{api_key[:4]}...{api_key[-4:]}" if len(api_key) > 8 else "****"
return openai_error(f"Invalid API key: {masked}.", ...)
```

**After:** extracted to a helper near `extract_api_key`:
```python
def _mask_api_key(key: str) -> str:
    return f"{key[:4]}...{key[-4:]}" if len(key) > 8 else "****"

# both call sites:
return openai_error(f"Invalid API key: {_mask_api_key(api_key)}.", ...)
```

**Why:** Verbatim duplication of a policy expression. The masking logic is now a
single point of change.

---

## 3. DRY: Content-list flattening duplicated inside `messages_to_prompt`

**File:** `main.py` — `messages_to_prompt` (role `"tool"` branch and general branch)

**Before:** (identical three-line comprehension in two places)
```python
content = "\n".join(
    item.get("text", "")
    for item in content
    if isinstance(item, dict) and "text" in item
)
```

**After:** extracted to a helper:
```python
def _flatten_content(content: list) -> str:
    return "\n".join(
        item.get("text", "")
        for item in content
        if isinstance(item, dict) and "text" in item
    )

# both call sites:
content = _flatten_content(content)
```

**Why:** Same logic repeated for both `role="tool"` messages and all other roles.
One definition, two uses.

---

## 4. Redundant `tools or None` normalization at call sites

**File:** `main.py` — `chat_completions` route

**Before:**
```python
result = transform_nonstream(resp.json(), model, prompt_tokens, tools or None)
...
stream_response(stream_resp, model, prompt_tokens, tools or None)
```

**After:**
```python
result = transform_nonstream(resp.json(), model, prompt_tokens, tools)
...
stream_response(stream_resp, model, prompt_tokens, tools)
```

**Why:** `tools` is already guaranteed to be a `list` at both call sites — either the
original list from the request body, or `[]` when `tool_choice == "none"`. Both
`transform_nonstream` and `stream_response` guard with `if tools:`, which evaluates
to `False` for both `[]` and `None`. The `or None` conversion bought nothing and
obscured the actual type.

---

## 5. Redundant state: `_cache_valid` flag in `ModelRegistry`

**File:** `model_registry.py`

**Before:** two separate fields tracking freshness:
```python
self._fetched_at  = 0.0
self._cache_valid = False
...
self._fetched_at  = time.monotonic()
self._cache_valid = True
...
if self._cache_valid and (time.monotonic() - self._fetched_at) < self._ttl:
    return
```

**After:** `_fetched_at` alone is sufficient:
```python
self._fetched_at = 0.0  # 0.0 = never fetched; positive = data available
...
self._fetched_at = time.monotonic()
...
if self._fetched_at and (time.monotonic() - self._fetched_at) < self._ttl:
    return
```

**Why:** `_fetched_at` starts at `0.0` (falsy) and becomes a positive monotonic
timestamp after the first successful fetch. `_cache_valid` tracked exactly the same
boolean (`_fetched_at > 0`) and had to be kept in sync on every write. With
`_fetched_at`, a single field encodes both "has data ever been fetched" and "when was
it fetched". The `_cache_valid = True` line in `_fetch_and_store` was dead weight.

---

## 6. Redundant allocation: `permitted_set` rebuilt on every call

**File:** `model_registry.py` — `get_available_models`

**Before:**
```python
if self._permitted:
    permitted_set = set(self._permitted)  # rebuilt on every call
    return [m for m in all_models if m in permitted_set]
```

**After:** pre-computed once in `__init__`:
```python
# __init__:
self._permitted_set = set(permitted)  # computed once

# get_available_models:
if self._permitted_set:
    return [m for m in all_models if m in self._permitted_set]
```

**Why:** `permitted` is passed at construction and never mutated. Creating a new
`set` on every call to `get_available_models` — which is invoked on every request
when `RESTRICT_MODELS=true` and on every `/v1/models` poll — was a gratuitous
allocation. Pre-computing it in `__init__` costs nothing.

---

## 7. Efficiency: 7 regex patterns compiled inline on every call

**File:** `function_calling.py` — `parse_tool_calls`

**Before:** all 7 patterns (plus the inner parameter pattern for the Anthropic XML
path) were compiled inline inside `re.finditer(pattern, text, re.DOTALL)` on each
invocation:
```python
for match in re.finditer(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", text, re.DOTALL):
    ...
for match in re.finditer(r'<invoke\s+name="([^"]+)">(.*?)</invoke>', text, re.DOTALL):
    for param in re.finditer(r'<parameter\s+name="([^"]+)">(.*?)</parameter>', ...):
```

**After:** pre-compiled as module-level constants:
```python
_RE_P1 = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)
_RE_P3_INVOKE = re.compile(r'<invoke\s+name="([^"]+)">(.*?)</invoke>', re.DOTALL)
_RE_P3_PARAM  = re.compile(r'<parameter\s+name="([^"]+)">(.*?)</parameter>', re.DOTALL)
# ... etc.

for match in _RE_P1.finditer(text):
```

**Why:** CPython's `re` module caches up to 512 compiled patterns, so repeated calls
with the same pattern string are not always catastrophically expensive — but each
call still incurs a dict hash lookup against the cache. More importantly, the inner
`_RE_P3_PARAM` pattern was compiled once per `<invoke>` match (i.e. per tool call in
legacy Anthropic XML format), not just once per `parse_tool_calls` call. Pre-compiling
at module level moves all compilation cost to import time and avoids the cache lookup
on every hot-path call.

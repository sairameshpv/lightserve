"""The only place that talks to a model server. Everything else calls `chat`, so serving the demo
through another engine (lightserve, later) means writing a second version of this one function.

This version: vLLM's OpenAI-style chat API (`/v1/chat/completions`), text in, text out.
"""
import json
import time
import urllib.error
import urllib.request


def chat(messages: list, model: str, base_url: str = "http://localhost:8000", temperature: float = 0.0,
         top_p: float = 1.0, seed=None, stop=None, max_tokens: int = 512, timeout: float = 120) -> dict:
    """One reply -> {"text", "prompt_tokens", "completion_tokens", "seconds"}. Raises RuntimeError
    if the server answers with an error. `model` picks the model: the base model's name or "v3"."""
    body = {"model": model, "messages": messages, "temperature": temperature, "top_p": top_p, "max_tokens": max_tokens}
    if seed is not None:
        body["seed"] = seed
    if stop:
        body["stop"] = stop
    req = urllib.request.Request(f"{base_url}/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    start = time.time()
    for attempt in (1, 2):              # one retry if the connection is dropped before an answer
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                out = json.loads(resp.read())
            break
        except urllib.error.HTTPError as e:      # the server answered with an error: a real problem, no retry
            raise RuntimeError(f"{model}: server answered {e.code}: {e.read()[:200]!r}") from e
        except (ConnectionError, urllib.error.URLError) as e:
            if attempt == 2 or isinstance(getattr(e, "reason", None), TimeoutError):
                raise
            time.sleep(0.2)
    usage = out.get("usage") or {}
    return {"text": out["choices"][0]["message"]["content"] or "", "prompt_tokens": usage.get("prompt_tokens"),
            "completion_tokens": usage.get("completion_tokens"), "seconds": time.time() - start}

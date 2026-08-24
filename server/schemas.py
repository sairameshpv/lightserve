"""OpenAI-compatible request/response shapes for POST /v1/completions.

This repo has no tokenizer -- model/llm_engine.py's generate() docstring is
explicit that "this repo has no tokenizer... callers own tokenization".
Rather than inventing a bespoke, non-OpenAI request shape to route around
that gap, `prompt` here uses OpenAI's *own* array-of-token-ids completions
mode (the real API accepts a string, a list of strings, a list of tokens, or
a list of token-id lists for `prompt` -- this is the one variant of theirs
that needs no string encode/decode at all). A plain string prompt is
therefore not accepted; a caller with a real tokenizer encodes client-side,
same as it would have to decode `token_ids` back to text client-side (see
CompletionChoice below).

`text` in every response is consequently *not* decoded text -- there's no
vocabulary to decode against (model/minimal_llama.py's weights are random,
per its own module docstring). It's a space-joined string of the token ids,
kept only so curl output and any real OpenAI-client's `.text` accessor stay
human-legible rather than empty; `token_ids` is the field a real caller
should actually use.
"""
from typing import Optional

from pydantic import BaseModel, Field


class CompletionRequest(BaseModel):
    model: str = "lightserve"
    prompt: list[int] = Field(
        ..., description="Token ids (OpenAI's array-of-tokens prompt mode) -- see module docstring.",
    )
    max_tokens: int = 16
    eos_token_id: Optional[int] = None  # no fixed tokenizer means no fixed eos convention to default to
    stream: bool = False
    timeout_s: Optional[float] = Field(
        None, description="Overrides the server's default per-request timeout; see engine_worker.py.",
    )
    request_id: Optional[str] = None


class CompletionChoice(BaseModel):
    index: int = 0
    text: str
    token_ids: list[int]
    finish_reason: Optional[str] = None


class CompletionResponse(BaseModel):
    id: str
    object: str = "text_completion"
    model: str
    choices: list[CompletionChoice]


class CompletionStreamChoice(BaseModel):
    index: int = 0
    text: str
    token_ids: list[int]
    finish_reason: Optional[str] = None


class CompletionStreamChunk(BaseModel):
    """One SSE frame's JSON payload -- one per newly generated token, plus a
    final frame carrying `finish_reason` and no new tokens. See app.py's
    _sse_stream.
    """
    id: str
    object: str = "text_completion.chunk"
    model: str
    choices: list[CompletionStreamChoice]

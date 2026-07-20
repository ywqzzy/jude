"""jude.ai — high-level AI function APIs."""

from .jude import ai as _ai

embed_text = _ai.embed_text
classify_text = _ai.classify_text
prompt = _ai.prompt
embed = _ai.embed
load_provider = _ai.load_provider
get_token_metrics = _ai.get_token_metrics
reset_token_metrics = _ai.reset_token_metrics

__all__ = [
    "embed_text",
    "classify_text",
    "prompt",
    "embed",
    "load_provider",
    "get_token_metrics",
    "reset_token_metrics",
]

"""Daft bridge — run Daft DataFrame operations (multimodal decode/resize,
url.download, embed_text / embed_image / classify_image, ...) on jude data and
bring the result back, all zero-copy through Arrow.

jude already has its own `.image`/`.url` expression layer and (via Lance) vector
search; Daft's complementary value is its embedding + model ops. The natural
pipeline: jude relation -> Daft (embed_text/embed_image) -> jude relation ->
write_lance + vector index -> ANN search.
"""

from __future__ import annotations

from typing import Any, Callable

import pyarrow as pa


def to_daft(table: pa.Table) -> Any:
    import daft

    return daft.from_arrow(table)


def from_daft(df: Any) -> pa.Table:
    # A Daft DataFrame (or anything exposing to_arrow).
    return df.to_arrow()


def transform(table: pa.Table, fn: Callable[[Any], Any]) -> pa.Table:
    """Apply a user function `fn(daft.DataFrame) -> daft.DataFrame` to the data
    and return Arrow. Gives full access to Daft's expression API (multimodal,
    embeddings, model inference) without jude wrapping each op."""
    import daft

    df = daft.from_arrow(table)
    out = fn(df)
    if not hasattr(out, "to_arrow"):
        raise TypeError("daft transform must return a daft.DataFrame")
    return out.to_arrow()

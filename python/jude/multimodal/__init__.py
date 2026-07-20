"""jude.multimodal — decoders for encoded multimodal columns.

The multimodal *logical types* live in ``jude.types`` (TensorType / Image /
Audio / Video / Document). This package holds the *decode ops* that turn the
encoded-bytes columns produced by ``jude.sources`` into tensors / samples / text
you can query. Each decoder is a batch-in / batch-out callable, so it plugs
straight into ``Relation.map_batches`` / ``Relation.flat_map`` and into
``jude.pipeline`` stages (see ``DecodeStage``).
"""

from __future__ import annotations

from jude.multimodal.decoders import (
    decode_audio_batch,
    decode_document_batch,
    decode_image_batch,
    decode_video_batch,
)

__all__ = [
    "decode_image_batch",
    "decode_audio_batch",
    "decode_video_batch",
    "decode_document_batch",
]

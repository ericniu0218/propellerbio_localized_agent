"""Byte-level compression, behind one small seam.

Prefers ``zstandard`` because it is several times faster than the standard
library's ``lzma`` at similar ratios, and the data this store holds - logs,
CSV rows, source code - is exactly the repetitive kind it does best on. It
falls back to ``lzma`` when zstandard is not installed, so a bare checkout
still works and the test suite never depends on an optional package.

The codec name is written into every artifact's manifest, so a store created
on a machine that had zstandard still reads correctly on one that does not,
as long as that machine can decompress it. Reading an artifact whose codec is
unavailable raises a clear error rather than returning garbage.
"""

from __future__ import annotations

import lzma

ZSTD = "zstd"
LZMA = "lzma"

_ZSTD_LEVEL = 10
_LZMA_PRESET = 6

try:  # pragma: no cover - which branch runs depends on the environment
    import zstandard

    _HAVE_ZSTD = True
except ImportError:  # pragma: no cover
    zstandard = None
    _HAVE_ZSTD = False


class CodecUnavailable(RuntimeError):
    """Raised when stored data needs a codec this machine does not have."""


def default_codec() -> str:
    """The codec new artifacts are written with."""
    return ZSTD if _HAVE_ZSTD else LZMA


def compress(data: bytes, codec: str | None = None) -> tuple[bytes, str]:
    """Compress ``data``; returns the bytes and the codec actually used."""
    codec = codec or default_codec()
    if codec == ZSTD:
        if not _HAVE_ZSTD:
            raise CodecUnavailable("zstandard is not installed")
        return zstandard.ZstdCompressor(level=_ZSTD_LEVEL).compress(data), ZSTD
    if codec == LZMA:
        return lzma.compress(data, preset=_LZMA_PRESET), LZMA
    raise CodecUnavailable(f"Unknown codec {codec!r}")


def decompress(data: bytes, codec: str) -> bytes:
    if codec == ZSTD:
        if not _HAVE_ZSTD:
            raise CodecUnavailable(
                "This artifact was stored with zstandard, which is not installed here. "
                "Install it with `pip install zstandard`."
            )
        return zstandard.ZstdDecompressor().decompress(data)
    if codec == LZMA:
        return lzma.decompress(data)
    raise CodecUnavailable(f"Unknown codec {codec!r}")

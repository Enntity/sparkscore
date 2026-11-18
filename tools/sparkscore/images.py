# SPDX-License-Identifier: AGPL-3.0-only
"""Image inputs: base64 data URLs as d3-ultra's own server takes them (PNG, JPEG or WebP), read only as far as
their headers. The engine decodes the pixels; here each image's size gives its count of input tokens, which the
prompt must hold as image pads (one per 32 x 32 pixels after d3's processor resizes it)."""

import base64
import binascii
import math
import struct
from dataclasses import dataclass

from tools.sparkscore.render import Unsupported

MAX_IMAGES = 8
MAX_IMAGE_BYTES = 8_000_000  # Atlas's limit on one encoded image
MAX_SOURCE_PIXELS = 16_000_000
TYPES = {"png": "png", "jpeg": "jpeg", "jpg": "jpeg", "webp": "webp"}


@dataclass(frozen=True)
class Geometry:
    """d3-ultra's image processor: 16-pixel patches merged 2 x 2, images resized to 65,536..1,638,400 pixels."""

    patch: int = 16
    merge: int = 2
    min_pixels: int = 65536
    max_pixels: int = 1638400

    def target(self, width, height):
        """(height, width) after the processor's resize (Qwen's smart_resize)."""
        f = self.patch * self.merge
        if max(height, width) / min(height, width) > 200:
            raise Unsupported(f"an image's aspect ratio must be under 200, got {width}x{height}")
        h, w = max(f, round(height / f) * f), max(f, round(width / f) * f)
        if h * w > self.max_pixels:
            beta = math.sqrt(height * width / self.max_pixels)
            h, w = max(f, math.floor(height / beta / f) * f), max(f, math.floor(width / beta / f) * f)
        elif h * w < self.min_pixels:
            beta = math.sqrt(self.min_pixels / (height * width))
            h, w = math.ceil(height * beta / f) * f, math.ceil(width * beta / f) * f
        return h, w

    def pads(self, width, height):
        h, w = self.target(width, height)
        f = self.patch * self.merge
        return (h // f) * (w // f)


@dataclass(frozen=True)
class Image:
    format: str  # png, jpeg or webp
    data: bytes
    width: int
    height: int

    def wire(self):
        """The image as Atlas /v1/score takes it."""
        return {"data_base64": base64.b64encode(self.data).decode("ascii"), "format": self.format}


def parse(urls):
    """A request's `images`: a list of data:image/<png|jpeg|webp>;base64 URLs -> [Image]. ValueError says why not."""
    if not isinstance(urls, list):
        raise ValueError("images takes a list of base64 data URLs")
    if len(urls) > MAX_IMAGES:
        raise ValueError(f"a request takes at most {MAX_IMAGES} images, got {len(urls)}")
    return [_one(i, url) for i, url in enumerate(urls)]


def _one(i, url):
    if not isinstance(url, str) or not url.startswith("data:image/"):
        raise ValueError(f"image {i}: send a data:image/...;base64 URL (no paths or web addresses)")
    head, sep, payload = url.partition(",")
    kind = head[len("data:image/"):]
    if not sep or not kind.endswith(";base64") or kind[: -len(";base64")].lower() not in TYPES:
        raise ValueError(f"image {i}: take data:image/png, jpeg or webp;base64 URLs")
    if len(payload) > (MAX_IMAGE_BYTES + 2) // 3 * 4:
        raise ValueError(f"image {i}: over the {MAX_IMAGE_BYTES:,} byte limit")
    try:
        data = base64.b64decode(payload, validate=True)
    except (binascii.Error, ValueError) as error:
        raise ValueError(f"image {i}: invalid base64 ({error})") from None
    fmt = _sniff(data)
    if fmt is None:
        raise ValueError(f"image {i}: not a PNG, JPEG or WebP file")
    width, height = dimensions(fmt, data)
    if not (width and height) or width * height > MAX_SOURCE_PIXELS:
        raise ValueError(f"image {i}: {width}x{height} pixels; at most {MAX_SOURCE_PIXELS:,} are accepted")
    return Image(fmt, data, width, height)


def _sniff(data):
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "png"
    if data.startswith(b"\xff\xd8"):
        return "jpeg"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "webp"
    return None


def dimensions(fmt, data):
    """(width, height) from the file header. ValueError when the header cannot be read."""
    try:
        if fmt == "png":
            if data[12:16] != b"IHDR":
                raise ValueError("no IHDR")
            return struct.unpack(">II", data[16:24])
        if fmt == "jpeg":
            return _jpeg_size(data)
        return _webp_size(data)
    except (struct.error, IndexError, ValueError) as error:
        raise ValueError(f"unreadable {fmt} header ({error})") from None


def _jpeg_size(data):
    i = 2
    while i + 4 <= len(data):
        if data[i] != 0xFF:
            raise ValueError("bad marker")
        marker = data[i + 1]
        if marker == 0xFF:  # fill byte
            i += 1
            continue
        if marker in (0xD8, 0x01) or 0xD0 <= marker <= 0xD7:
            i += 2
            continue
        (length,) = struct.unpack(">H", data[i + 2 : i + 4])
        if 0xC0 <= marker <= 0xCF and marker not in (0xC4, 0xC8, 0xCC):
            height, width = struct.unpack(">HH", data[i + 5 : i + 9])
            return width, height
        i += 2 + length
    raise ValueError("no frame header")


def _webp_size(data):
    chunk = data[12:16]
    if chunk == b"VP8 ":
        width, height = struct.unpack("<HH", data[26:30])
        return width & 0x3FFF, height & 0x3FFF
    if chunk == b"VP8L":
        b = data[21:25]
        return 1 + (((b[1] & 0x3F) << 8) | b[0]), 1 + (((b[3] & 0xF) << 10) | (b[2] << 2) | ((b[1] & 0xC0) >> 6))
    if chunk == b"VP8X":
        return 1 + int.from_bytes(data[24:27], "little"), 1 + int.from_bytes(data[27:30], "little")
    raise ValueError("unknown WebP chunk")

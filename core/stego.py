from __future__ import annotations

from typing import Iterable, Iterator, List, Sequence, Tuple

from PIL import Image


SUPPORTED_METHODS = {"sequential", "interleaved"}


def xor_with_password(data: bytes, password: str) -> bytes:
    if not password:
        return data
    pw = password.encode("utf-8")
    return bytes(b ^ pw[i % len(pw)] for i, b in enumerate(data))


def get_carrier_order(bits_per_channel: int, method: str) -> List[Tuple[int, int]]:
    validate_bits_and_method(bits_per_channel, method)
    if method == "interleaved":
        return [(channel, bit_idx) for bit_idx in range(bits_per_channel) for channel in range(3)]
    return [(channel, bit_idx) for channel in range(3) for bit_idx in range(bits_per_channel)]


def validate_bits_and_method(bits_per_channel: int, method: str) -> None:
    if bits_per_channel not in (1, 2, 3):
        raise ValueError("Поддерживаются только 1, 2 или 3 бита на канал")
    if method not in SUPPORTED_METHODS:
        raise ValueError(f"Неизвестный метод встраивания: {method}")


def max_message_bytes(image_size: Sequence[int], bits_per_channel: int) -> int:
    validate_bits_and_method(bits_per_channel, "sequential")
    width, height = int(image_size[0]), int(image_size[1])
    total_bytes = (width * height * 3 * bits_per_channel) // 8
    return max(0, total_bytes - 4)


def prepare_payload(message: str, password: str) -> bytes:
    plain = message.encode("utf-8")
    protected = xor_with_password(plain, password)
    return len(protected).to_bytes(4, "big") + protected


def decode_payload(payload: bytes, password: str) -> str:
    if len(payload) < 4:
        raise ValueError("Мало данных")
    msg_len = int.from_bytes(payload[:4], "big")
    msg_bytes = payload[4 : 4 + msg_len]
    if len(msg_bytes) != msg_len:
        raise ValueError("Неполное сообщение")
    decrypted = xor_with_password(msg_bytes, password)
    return decrypted.decode("utf-8", errors="replace")


def encode_payload_into_image(
    image: Image.Image, payload: bytes, bits_per_channel: int, method: str
) -> Image.Image:
    validate_bits_and_method(bits_per_channel, method)
    if image.mode != "RGB":
        image = image.convert("RGB")

    total_bits_capacity = image.size[0] * image.size[1] * 3 * bits_per_channel
    payload_bits = len(payload) * 8
    if payload_bits > total_bits_capacity:
        raise ValueError(f"Сообщение слишком большое ({payload_bits} бит > {total_bits_capacity})")

    bit_stream = _bytes_to_bits(payload)
    out = image.copy()
    pixels_in = image.load()
    pixels_out = out.load()
    w, h = image.size
    carrier_order = get_carrier_order(bits_per_channel, method)

    idx = 0
    for y in range(h):
        for x in range(w):
            channels = list(pixels_in[x, y])
            for channel_idx, bit_idx in carrier_order:
                if idx >= len(bit_stream):
                    break
                bit = bit_stream[idx]
                if bit:
                    channels[channel_idx] |= (1 << bit_idx)
                else:
                    channels[channel_idx] &= ~(1 << bit_idx)
                idx += 1
            pixels_out[x, y] = tuple(channels)
            if idx >= len(bit_stream):
                return out
    return out


def decode_payload_from_image(image: Image.Image, bits_per_channel: int, method: str) -> bytes:
    validate_bits_and_method(bits_per_channel, method)
    if image.mode != "RGB":
        image = image.convert("RGB")

    bits_iter = _iter_bits_from_image(image, bits_per_channel, method)

    header_bits = [next(bits_iter, None) for _ in range(32)]
    if any(bit is None for bit in header_bits):
        raise ValueError("Нет заголовка")
    msg_len = _bits_to_int(header_bits)

    max_bytes = max_message_bytes(image.size, bits_per_channel)
    if msg_len > max_bytes:
        raise ValueError("Некорректная длина сообщения")

    payload_bits_count = msg_len * 8
    msg_bits = [next(bits_iter, None) for _ in range(payload_bits_count)]
    if any(bit is None for bit in msg_bits):
        raise ValueError("Неполное сообщение")

    msg_bytes = _bits_to_bytes(msg_bits)
    return msg_len.to_bytes(4, "big") + msg_bytes


def encode_text_into_image(
    image: Image.Image, message: str, password: str, bits_per_channel: int, method: str
) -> Image.Image:
    payload = prepare_payload(message, password)
    return encode_payload_into_image(image, payload, bits_per_channel, method)


def decode_text_from_image(
    image: Image.Image, password: str, bits_per_channel: int, method: str
) -> str:
    payload = decode_payload_from_image(image, bits_per_channel, method)
    return decode_payload(payload, password)


def _bytes_to_bits(data: bytes) -> List[int]:
    bits: List[int] = []
    for byte in data:
        for shift in range(7, -1, -1):
            bits.append((byte >> shift) & 1)
    return bits


def _bits_to_int(bits: Sequence[int]) -> int:
    value = 0
    for bit in bits:
        value = (value << 1) | int(bit)
    return value


def _bits_to_bytes(bits: Sequence[int]) -> bytes:
    out = bytearray()
    for start in range(0, len(bits), 8):
        chunk = bits[start : start + 8]
        if len(chunk) < 8:
            break
        out.append(_bits_to_int(chunk))
    return bytes(out)


def _iter_bits_from_image(image: Image.Image, bits_per_channel: int, method: str) -> Iterator[int]:
    pixels = image.load()
    w, h = image.size
    carrier_order = get_carrier_order(bits_per_channel, method)
    for y in range(h):
        for x in range(w):
            channels = pixels[x, y]
            for channel_idx, bit_idx in carrier_order:
                yield (channels[channel_idx] >> bit_idx) & 1


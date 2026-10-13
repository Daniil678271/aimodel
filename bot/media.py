"""Работа с вложениями: подготовка изображений для Gemini."""
from __future__ import annotations

import io
from dataclasses import dataclass

from PIL import Image, ImageOps, UnidentifiedImageError

from errors import MediaError

MAX_IMAGE_SIDE = 2048  # достаточно для распознавания мелкого текста, но экономит трафик и токены
PDF_MIME = "application/pdf"


@dataclass(frozen=True, slots=True)
class Media:
    data: bytes
    mime_type: str

    @property
    def is_pdf(self) -> bool:
        return self.mime_type == PDF_MIME


def prepare_image(raw: bytes) -> bytes:
    """Проверяет изображение, исправляет поворот по EXIF, уменьшает и сохраняет как JPEG.

    Вызывать через asyncio.to_thread — операция CPU-bound.
    """
    try:
        with Image.open(io.BytesIO(raw)) as source:
            image = ImageOps.exif_transpose(source)
            has_alpha = image.mode in ("RGBA", "LA") or (
                image.mode == "P" and "transparency" in image.info
            )
            if has_alpha:
                rgba = image.convert("RGBA")
                background = Image.new("RGB", rgba.size, (255, 255, 255))
                background.paste(rgba, mask=rgba.split()[-1])
                image = background
            else:
                image = image.convert("RGB")

            if max(image.size) > MAX_IMAGE_SIDE:
                image.thumbnail((MAX_IMAGE_SIDE, MAX_IMAGE_SIDE), Image.Resampling.LANCZOS)

            output = io.BytesIO()
            image.save(output, format="JPEG", quality=92, optimize=True)
            return output.getvalue()
    except (UnidentifiedImageError, Image.DecompressionBombError, OSError, ValueError):
        raise MediaError(
            "Не удалось прочитать изображение. Поддерживаются JPEG, PNG, WEBP, GIF, BMP. "
            "Попробуйте отправить файл как обычное фото или в формате JPEG/PNG."
        ) from None

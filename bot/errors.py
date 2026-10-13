"""Исключения с текстом, который можно безопасно показать пользователю."""
from __future__ import annotations


class BotError(Exception):
    """Базовая ошибка: user_message показывается пользователю в чате."""

    def __init__(self, user_message: str) -> None:
        super().__init__(user_message)
        self.user_message = user_message


class GeminiError(BotError):
    """Ошибка при обращении к Gemini API."""


class MediaError(BotError):
    """Проблема с присланным файлом или изображением."""

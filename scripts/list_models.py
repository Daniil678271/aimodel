"""Показывает модели Gemini, доступные вашему API-ключу и поддерживающие generateContent.

Запуск:  python -m scripts.list_models
Docker:  docker compose run --rm gemini-bot python -m scripts.list_models
"""
from __future__ import annotations

import os
import sys

from dotenv import load_dotenv
from google import genai


def main() -> None:
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key:
        sys.exit("GEMINI_API_KEY не задан (впишите его в .env)")

    client = genai.Client(api_key=api_key)
    found = 0
    for model in client.models.list():
        actions = getattr(model, "supported_actions", None) or []
        if "generateContent" not in actions:
            continue
        name = (model.name or "").removeprefix("models/")
        print(f"{name:45} {getattr(model, 'display_name', '') or ''}")
        found += 1

    if not found:
        print("Подходящих моделей не найдено. Проверьте ключ и доступ к Gemini API.")


if __name__ == "__main__":
    main()

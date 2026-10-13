"""Загрузка и проверка настроек из переменных окружения (.env)."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


class ConfigError(RuntimeError):
    """Ошибка конфигурации с понятным сообщением для администратора."""


def _env(name: str, default: str = "") -> str:
    return os.getenv(name, default).strip()


def _int(name: str, default: int, minimum: int = 0) -> int:
    raw = _env(name)
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError:
        raise ConfigError(f"{name} должен быть целым числом, сейчас: {raw!r}") from None
    if value < minimum:
        raise ConfigError(f"{name} должен быть не меньше {minimum}, сейчас: {value}")
    return value


def _float_optional(name: str) -> float | None:
    raw = _env(name)
    if not raw:
        return None
    try:
        return float(raw.replace(",", "."))
    except ValueError:
        raise ConfigError(f"{name} должен быть числом, сейчас: {raw!r}") from None


def _bool(name: str, default: bool = False) -> bool:
    raw = _env(name).lower()
    if not raw:
        return default
    return raw in {"1", "true", "yes", "y", "on", "да"}


def _id_set(name: str) -> frozenset[int]:
    raw = _env(name)
    ids: set[int] = set()
    for chunk in raw.replace(";", ",").split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if not chunk.lstrip("-").isdigit():
            raise ConfigError(f"{name}: {chunk!r} не похоже на Telegram ID (нужны только числа)")
        ids.add(int(chunk))
    return frozenset(ids)


@dataclass(frozen=True, slots=True)
class Settings:
    telegram_token: str
    gemini_api_key: str
    gemini_model: str
    gemini_fallback_model: str
    gemini_temperature: float | None
    gemini_max_output_tokens: int
    gemini_request_timeout: int
    gemini_use_search: bool
    max_history_messages: int
    max_file_mb: int
    rate_limit_per_minute: int
    daily_limit_per_user: int
    allowed_user_ids: frozenset[int]
    database_path: Path
    log_level: str


def load_settings() -> Settings:
    token = _env("TELEGRAM_BOT_TOKEN")
    if not token:
        raise ConfigError(
            "Не задан TELEGRAM_BOT_TOKEN. Получите токен у @BotFather и впишите его в файл .env"
        )
    if ":" not in token:
        raise ConfigError("TELEGRAM_BOT_TOKEN выглядит неверно (ожидается формат 123456789:AA...)")

    api_key = _env("GEMINI_API_KEY")
    if not api_key:
        raise ConfigError(
            "Не задан GEMINI_API_KEY. Создайте ключ на https://aistudio.google.com/apikey "
            "и впишите его в файл .env"
        )

    model = _env("GEMINI_MODEL", "gemini-3.6-flash")
    if not model:
        raise ConfigError("GEMINI_MODEL не должен быть пустым")

    log_level = _env("LOG_LEVEL", "INFO").upper()
    if log_level not in {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}:
        raise ConfigError(f"LOG_LEVEL={log_level!r} не поддерживается")

    return Settings(
        telegram_token=token,
        gemini_api_key=api_key,
        gemini_model=model,
        gemini_fallback_model=_env("GEMINI_FALLBACK_MODEL", "gemini-3.5-flash"),
        gemini_temperature=_float_optional("GEMINI_TEMPERATURE"),
        gemini_max_output_tokens=_int("GEMINI_MAX_OUTPUT_TOKENS", 16384, minimum=256),
        gemini_request_timeout=_int("GEMINI_REQUEST_TIMEOUT", 120, minimum=10),
        gemini_use_search=_bool("GEMINI_USE_SEARCH", False),
        max_history_messages=_int("MAX_HISTORY_MESSAGES", 24, minimum=0),
        max_file_mb=_int("MAX_FILE_MB", 15, minimum=1),
        rate_limit_per_minute=_int("RATE_LIMIT_PER_MINUTE", 10, minimum=0),
        daily_limit_per_user=_int("DAILY_LIMIT_PER_USER", 0, minimum=0),
        allowed_user_ids=_id_set("ALLOWED_USER_IDS"),
        database_path=Path(_env("DATABASE_PATH", "data/bot.db")),
        log_level=log_level,
    )

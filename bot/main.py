"""Точка входа: python -m bot.main"""
from __future__ import annotations

import logging
import sys

from telegram import Update
from telegram.ext import Application

from .config import ConfigError, load_settings
from .gemini_client import GeminiService
from .handlers import BotHandlers
from .health import start_health_server
from .ratelimit import RateLimiter
from .storage import Storage


def configure_logging(level: str) -> None:
    logging.basicConfig(
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        level=getattr(logging, level),
    )
    # httpx на уровне INFO пишет в лог полные URL запросов, а в них — токен бота.
    for noisy in ("httpx", "httpcore", "hpack", "google_genai", "google.auth"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def main() -> None:
    try:
        settings = load_settings()
    except ConfigError as exc:
        sys.exit(f"Ошибка настройки: {exc}")

    configure_logging(settings.log_level)
    logger = logging.getLogger("bot")

    storage = Storage(settings.database_path, settings.max_history_messages)
    gemini = GeminiService(settings)
    limiter = RateLimiter(settings.rate_limit_per_minute)
    handlers = BotHandlers(settings, storage, gemini, limiter)

    health_server = None

    async def post_init(app: Application) -> None:
        nonlocal health_server
        # Сначала поднимаем порт (Render ждёт его), затем обычная инициализация бота.
        health_server = await start_health_server()
        await handlers.post_init(app)

    async def post_shutdown(app: Application) -> None:
        if health_server is not None:
            health_server.close()
        await handlers.post_shutdown(app)

    application = (
        Application.builder()
        .token(settings.telegram_token)
        .concurrent_updates(True)
        .connect_timeout(20)
        .read_timeout(30)
        .write_timeout(30)
        .pool_timeout(30)
        .post_init(post_init)
        .post_shutdown(post_shutdown)
        .build()
    )
    handlers.register(application)

    logger.info("Запуск. Модель: %s, резервная: %s", settings.gemini_model, settings.gemini_fallback_model or "—")
    application.run_polling(allowed_updates=Update.ALL_TYPES, drop_pending_updates=True)


if __name__ == "__main__":
    main()

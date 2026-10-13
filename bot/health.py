"""Крошечный HTTP-сервер для бесплатного Web Service на Render.

Render требует, чтобы сервис слушал порт из переменной окружения PORT, а внешний
пингер (UptimeRobot и т.п.) раз в 5–10 минут обращается к нему, чтобы сервис не засыпал.
Сервер отвечает "ok" на любой GET/HEAD-запрос и больше ничего не делает.
Если PORT не задан (локальный запуск, Docker на своём сервере) — сервер не запускается.
"""
from __future__ import annotations

import asyncio
import logging
import os

logger = logging.getLogger("bot.health")

_BODY = b"ok"
_HEADERS = (
    b"HTTP/1.1 200 OK\r\n"
    b"Content-Type: text/plain; charset=utf-8\r\n"
    b"Content-Length: " + str(len(_BODY)).encode() + b"\r\n"
    b"Connection: close\r\n\r\n"
)


async def _handle(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
    try:
        # Читаем только строку запроса; тело и заголовки не нужны.
        line = await asyncio.wait_for(reader.readline(), timeout=5)
        is_head = line.startswith(b"HEAD")
        writer.write(_HEADERS + (b"" if is_head else _BODY))
        await writer.drain()
    except (asyncio.TimeoutError, ConnectionError):
        pass
    finally:
        writer.close()


async def start_health_server() -> asyncio.AbstractServer | None:
    raw = os.getenv("PORT", "").strip()
    if not raw:
        return None
    try:
        port = int(raw)
    except ValueError:
        logger.warning("PORT=%r не число — health-сервер не запущен", raw)
        return None
    server = await asyncio.start_server(_handle, host="0.0.0.0", port=port)
    logger.info("Health-сервер слушает порт %d", port)
    return server

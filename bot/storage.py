"""Хранилище на SQLite: история диалогов, настройки пользователей, суточные лимиты.

Используется стандартный модуль sqlite3; все вызовы выполняются в потоках через
asyncio.to_thread, поэтому цикл событий бота не блокируется.
"""
from __future__ import annotations

import asyncio
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path

DEFAULT_STYLE = "detailed"
MAX_STORED_CHARS = 12_000  # ограничение длины одного сохранённого сообщения

_SCHEMA = """
CREATE TABLE IF NOT EXISTS messages (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id    INTEGER NOT NULL,
    role       TEXT    NOT NULL CHECK (role IN ('user', 'model')),
    content    TEXT    NOT NULL,
    created_at TEXT    NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_messages_user ON messages (user_id, id);

CREATE TABLE IF NOT EXISTS user_settings (
    user_id INTEGER PRIMARY KEY,
    style   TEXT NOT NULL DEFAULT 'detailed'
);

CREATE TABLE IF NOT EXISTS daily_usage (
    user_id INTEGER NOT NULL,
    day     TEXT    NOT NULL,
    count   INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (user_id, day)
);
"""


def today_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


class Storage:
    def __init__(self, path: Path, max_messages: int) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._max_messages = max_messages
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(str(path), check_same_thread=False, timeout=30)
        with self._lock:
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.executescript(_SCHEMA)
            self._conn.commit()

    # ---------------------------------------------------------------- history
    def _get_history(self, user_id: int) -> list[tuple[str, str]]:
        if self._max_messages <= 0:
            return []
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT role, content FROM (
                    SELECT id, role, content FROM messages
                    WHERE user_id = ? ORDER BY id DESC LIMIT ?
                ) ORDER BY id ASC
                """,
                (user_id, self._max_messages),
            ).fetchall()
        history = [(role, content) for role, content in rows]
        # Диалог для Gemini должен начинаться с реплики пользователя.
        while history and history[0][0] != "user":
            history.pop(0)
        return history

    def _add_exchange(self, user_id: int, user_text: str, model_text: str) -> None:
        if self._max_messages <= 0:
            return
        with self._lock, self._conn:
            self._conn.executemany(
                "INSERT INTO messages (user_id, role, content) VALUES (?, ?, ?)",
                [
                    (user_id, "user", user_text[:MAX_STORED_CHARS]),
                    (user_id, "model", model_text[:MAX_STORED_CHARS]),
                ],
            )
            self._conn.execute(
                """
                DELETE FROM messages
                WHERE user_id = ? AND id NOT IN (
                    SELECT id FROM messages WHERE user_id = ? ORDER BY id DESC LIMIT ?
                )
                """,
                (user_id, user_id, self._max_messages),
            )

    def _clear_history(self, user_id: int) -> None:
        with self._lock, self._conn:
            self._conn.execute("DELETE FROM messages WHERE user_id = ?", (user_id,))

    # --------------------------------------------------------------- settings
    def _get_style(self, user_id: int) -> str:
        with self._lock:
            row = self._conn.execute(
                "SELECT style FROM user_settings WHERE user_id = ?", (user_id,)
            ).fetchone()
        return row[0] if row else DEFAULT_STYLE

    def _set_style(self, user_id: int, style: str) -> None:
        with self._lock, self._conn:
            self._conn.execute(
                """
                INSERT INTO user_settings (user_id, style) VALUES (?, ?)
                ON CONFLICT(user_id) DO UPDATE SET style = excluded.style
                """,
                (user_id, style),
            )

    # ----------------------------------------------------------- daily limits
    def _hit_daily(self, user_id: int, day: str, limit: int) -> bool:
        with self._lock, self._conn:
            row = self._conn.execute(
                "SELECT count FROM daily_usage WHERE user_id = ? AND day = ?", (user_id, day)
            ).fetchone()
            if row and row[0] >= limit:
                return False
            self._conn.execute(
                """
                INSERT INTO daily_usage (user_id, day, count) VALUES (?, ?, 1)
                ON CONFLICT(user_id, day) DO UPDATE SET count = count + 1
                """,
                (user_id, day),
            )
            self._conn.execute("DELETE FROM daily_usage WHERE day < ?", (day,))
            return True

    # ------------------------------------------------------------- async API
    async def get_history(self, user_id: int) -> list[tuple[str, str]]:
        return await asyncio.to_thread(self._get_history, user_id)

    async def add_exchange(self, user_id: int, user_text: str, model_text: str) -> None:
        await asyncio.to_thread(self._add_exchange, user_id, user_text, model_text)

    async def clear_history(self, user_id: int) -> None:
        await asyncio.to_thread(self._clear_history, user_id)

    async def get_style(self, user_id: int) -> str:
        return await asyncio.to_thread(self._get_style, user_id)

    async def set_style(self, user_id: int, style: str) -> None:
        await asyncio.to_thread(self._set_style, user_id, style)

    async def hit_daily(self, user_id: int, limit: int) -> bool:
        """Засчитывает запрос. Возвращает False, если суточный лимит уже исчерпан."""
        return await asyncio.to_thread(self._hit_daily, user_id, today_utc(), limit)

    def close(self) -> None:
        with self._lock:
            self._conn.close()

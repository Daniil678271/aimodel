"""Ограничение частоты запросов (скользящее окно) на одного пользователя."""
from __future__ import annotations

import time
from collections import defaultdict, deque


class RateLimiter:
    def __init__(self, max_calls: int, period: float = 60.0) -> None:
        self._max_calls = max_calls
        self._period = period
        self._calls: dict[int, deque[float]] = defaultdict(deque)

    def check(self, user_id: int) -> float:
        """Засчитывает обращение. Возвращает 0, если можно, иначе — сколько секунд подождать."""
        if self._max_calls <= 0:
            return 0.0

        now = time.monotonic()
        window = self._calls[user_id]
        while window and now - window[0] >= self._period:
            window.popleft()

        if len(window) >= self._max_calls:
            return max(0.1, self._period - (now - window[0]))

        window.append(now)
        return 0.0

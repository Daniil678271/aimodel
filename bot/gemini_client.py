"""Обёртка над официальным Google Gen AI SDK (пакет google-genai)."""
from __future__ import annotations

import asyncio
import logging
import random
from datetime import datetime, timezone

import httpx
from google import genai
from google.genai import errors, types

from .config import Settings
from .errors import GeminiError
from .media import Media

logger = logging.getLogger(__name__)

RETRYABLE_CODES = {429, 500, 502, 503, 504}
MAX_ATTEMPTS = 3

BASE_SYSTEM_PROMPT = """\
Ты — универсальный ИИ-ассистент в Telegram, аналог приложения Gemini. Ты помогаешь с учёбой, \
работой, программированием, текстами и повседневными вопросами: математика, физика, химия, \
биология, история, география, русский язык, литература, информатика и любые другие темы.

Правила:
1. Отвечай на языке пользователя (по умолчанию — на русском), если он не попросил иначе.
2. Если пользователь прислал изображение с задачей, примером, уравнением, тестом или заданием:
   - сначала кратко перепиши условие так, как ты его прочитал (чтобы пользователь мог заметить \
ошибку распознавания);
   - реши задачу по шагам и проверь результат;
   - в конце отдельной строкой выдели итог: **Ответ:** ...
   Если заданий несколько — решай каждое под своим номером.
3. Если часть изображения нечитаема или условие неоднозначно — прямо скажи, что именно не видно, \
и не выдумывай недостающие данные.
4. В тестах с вариантами ответа укажи правильный вариант и коротко объясни, почему он верный.
5. Если пользователь прислал текст, документ, график, таблицу или схему — распознай содержимое \
и выполни то, о чём он просит; без явной просьбы кратко объясни, что там изображено.
6. НЕ используй LaTeX и символы $. Формулы пиши обычным текстом и символами Юникода: x², x₁, √, \
≤, ≥, ≠, ×, ÷, π, ∑, ∫, Δ; дроби — как (a+b)/(c+d).
7. Форматирование должно подходить для Telegram: **жирный**, *курсив*, `код`, блоки кода с \
указанием языка (```python), списки через «- » или нумерацию. Не используй таблицы Markdown и HTML. \
Заголовки применяй редко.
8. Будь честным: если не уверен — скажи об этом; не выдумывай факты, цитаты и источники.
9. Учитывай контекст предыдущих сообщений диалога. Пользователь не видит ваш внутренний процесс \
рассуждений, поэтому в ответе давай только результат.
"""

STYLE_PROMPTS: dict[str, str] = {
    "detailed": (
        "Режим ответа: ПОДРОБНЫЙ. Для учебных задач давай пошаговое решение с пояснением каждого "
        "шага и итоговым ответом, чтобы пользователь понял ход мысли. Для остальных вопросов "
        "отвечай развёрнуто, но без лишней воды."
    ),
    "brief": (
        "Режим ответа: КРАТКИЙ. Давай итоговый ответ и самое необходимое пояснение (1–5 строк). "
        "Подробное решение приводи, только если пользователь попросит."
    ),
}

STYLE_LABELS: dict[str, str] = {
    "detailed": "📖 Подробно",
    "brief": "⚡ Кратко",
}


def build_system_prompt(style: str) -> str:
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    style_prompt = STYLE_PROMPTS.get(style, STYLE_PROMPTS["detailed"])
    return f"{BASE_SYSTEM_PROMPT}\n{style_prompt}\nСегодняшняя дата (UTC): {today}."


class _SwitchModel(Exception):
    """Внутренний сигнал: основная модель не справилась, можно пробовать резервную."""

    def __init__(self, error: GeminiError) -> None:
        super().__init__(error.user_message)
        self.error = error


def _translate_api_error(exc: errors.APIError, model: str) -> GeminiError:
    code = exc.code
    message = str(exc.message or "").lower()

    if code == 429:
        return GeminiError(
            "Достигнут лимит запросов Gemini API (на бесплатном тарифе он ограничен по запросам "
            "в минуту и в сутки). Подождите немного и повторите запрос."
        )
    if code in (401, 403):
        if "location" in message or "not supported" in message:
            return GeminiError(
                "Gemini API недоступен из региона, где запущен бот. "
                "Администратору: разместите бота на сервере в поддерживаемой стране."
            )
        return GeminiError(
            "Gemini API отклонил запрос: ключ недействителен или доступ запрещён. "
            "Администратору: проверьте GEMINI_API_KEY."
        )
    if code == 400:
        if "api key" in message or "api_key" in message:
            return GeminiError(
                "Gemini API не принял ключ. Администратору: проверьте GEMINI_API_KEY."
            )
        return GeminiError(
            "Gemini не смог обработать запрос: возможно, файл слишком большой или имеет "
            "неподдерживаемый формат. Попробуйте другой файл или сократите запрос."
        )
    if code == 404:
        return GeminiError(
            f"Модель «{model}» не найдена или больше недоступна. "
            "Администратору: укажите актуальную модель в GEMINI_MODEL "
            "(список: python -m scripts.list_models)."
        )
    if code is not None and code >= 500:
        return GeminiError(
            "Сервис Gemini временно перегружен или недоступен. Повторите запрос через минуту."
        )
    return GeminiError(f"Ошибка Gemini API (код {code}). Попробуйте ещё раз позже.")


class GeminiService:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._client = genai.Client(
            api_key=settings.gemini_api_key,
            http_options=types.HttpOptions(timeout=settings.gemini_request_timeout * 1000),
        )

    # ------------------------------------------------------------------ public
    async def ping(self) -> None:
        """Проверка ключа и модели при запуске (не тратит токены)."""
        model = self._settings.gemini_model
        try:
            info = await asyncio.wait_for(self._client.aio.models.get(model=model), timeout=20)
            logger.info("Gemini API доступен. Модель: %s", getattr(info, "name", model))
        except errors.APIError as exc:
            logger.error(
                "Проверка Gemini API не пройдена: HTTP %s — %s. Бот запущен, но запросы могут "
                "завершаться ошибкой: проверьте GEMINI_API_KEY и GEMINI_MODEL.",
                exc.code,
                exc.message,
            )
        except Exception as exc:  # noqa: BLE001 — проверка не должна ронять запуск
            logger.warning("Не удалось проверить Gemini API при запуске: %s", exc)

    async def generate(
        self,
        history: list[tuple[str, str]],
        prompt: str,
        media: list[Media],
        style: str,
    ) -> str:
        contents = self._build_contents(history, prompt, media)
        config = self._build_config(style)

        models = [self._settings.gemini_model]
        fallback = self._settings.gemini_fallback_model
        if fallback and fallback != models[0]:
            models.append(fallback)

        last_error: GeminiError | None = None
        for index, model in enumerate(models):
            try:
                return await self._call_model(model, contents, config)
            except _SwitchModel as switch:
                last_error = switch.error
                if index + 1 < len(models):
                    logger.warning("Модель %s недоступна, пробую резервную %s", model, models[index + 1])
        raise last_error or GeminiError("Не удалось получить ответ от Gemini. Попробуйте позже.")

    # ----------------------------------------------------------------- private
    @staticmethod
    def _build_contents(
        history: list[tuple[str, str]], prompt: str, media: list[Media]
    ) -> list[types.Content]:
        contents = [
            types.Content(role=role, parts=[types.Part.from_text(text=text)])
            for role, text in history
        ]
        parts = [types.Part.from_bytes(data=item.data, mime_type=item.mime_type) for item in media]
        parts.append(types.Part.from_text(text=prompt))
        contents.append(types.Content(role="user", parts=parts))
        return contents

    def _build_config(self, style: str) -> types.GenerateContentConfig:
        kwargs: dict = {
            "system_instruction": build_system_prompt(style),
            "max_output_tokens": self._settings.gemini_max_output_tokens,
        }
        if self._settings.gemini_temperature is not None:
            kwargs["temperature"] = self._settings.gemini_temperature
        if self._settings.gemini_use_search:
            kwargs["tools"] = [types.Tool(google_search=types.GoogleSearch())]
        return types.GenerateContentConfig(**kwargs)

    async def _call_model(
        self, model: str, contents: list[types.Content], config: types.GenerateContentConfig
    ) -> str:
        last: GeminiError | None = None
        for attempt in range(1, MAX_ATTEMPTS + 1):
            try:
                response = await asyncio.wait_for(
                    self._client.aio.models.generate_content(
                        model=model, contents=contents, config=config
                    ),
                    timeout=self._settings.gemini_request_timeout + 15,
                )
                return self._extract_text(response)
            except errors.APIError as exc:
                error = _translate_api_error(exc, model)
                if exc.code == 404:
                    raise _SwitchModel(error) from exc
                if exc.code not in RETRYABLE_CODES:
                    raise error from exc
                logger.warning(
                    "Gemini %s: HTTP %s (попытка %d/%d)", model, exc.code, attempt, MAX_ATTEMPTS
                )
                last = error
            except (httpx.HTTPError, asyncio.TimeoutError) as exc:
                logger.warning(
                    "Gemini %s: сетевая ошибка %s (попытка %d/%d)",
                    model,
                    type(exc).__name__,
                    attempt,
                    MAX_ATTEMPTS,
                )
                last = GeminiError(
                    "Gemini не ответил вовремя или пропало соединение. Повторите запрос."
                )
            if attempt < MAX_ATTEMPTS:
                await asyncio.sleep(2**attempt + random.random())
        raise _SwitchModel(last or GeminiError("Gemini временно недоступен."))

    @staticmethod
    def _extract_text(response: types.GenerateContentResponse) -> str:
        try:
            text = response.text
        except Exception:  # noqa: BLE001 — SDK может бросить ошибку при нетекстовых частях
            text = None

        candidates = response.candidates or []
        finish = candidates[0].finish_reason if candidates else None
        finish_name = getattr(finish, "name", str(finish)) if finish is not None else ""

        if text and text.strip():
            text = text.strip()
            if finish_name == "MAX_TOKENS":
                text += "\n\n_(Ответ обрезан по лимиту длины — напишите «продолжи».)_"
            return text

        block = getattr(getattr(response, "prompt_feedback", None), "block_reason", None)
        if block is not None or finish_name in {"SAFETY", "PROHIBITED_CONTENT", "BLOCKLIST", "SPII"}:
            raise GeminiError(
                "Запрос или ответ заблокирован фильтрами безопасности Google. "
                "Попробуйте переформулировать вопрос."
            )
        if finish_name == "MAX_TOKENS":
            raise GeminiError(
                "Модель израсходовала лимит длины, не успев дать ответ. "
                "Попробуйте упростить запрос или разбить его на части."
            )
        if finish_name == "RECITATION":
            raise GeminiError("Ответ не может быть выдан из-за ограничений на цитирование.")
        raise GeminiError("Модель вернула пустой ответ. Попробуйте переформулировать запрос.")

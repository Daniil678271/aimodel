"""Обработчики Telegram: команды, текст, изображения, альбомы, кнопки меню, ошибки."""
from __future__ import annotations

import asyncio
import contextlib
import html
import logging
import math
from collections import defaultdict
from datetime import timedelta
from typing import Any

from telegram import (
    Bot,
    BotCommand,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    LinkPreviewOptions,
    Message,
    ReplyKeyboardMarkup,
    Update,
    User,
)
from telegram.constants import ChatAction, ParseMode
from telegram.error import BadRequest, NetworkError, RetryAfter, TimedOut
from telegram.ext import (
    Application,
    CallbackQueryHandler,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

from .config import Settings
from .errors import BotError, MediaError
from .gemini_client import STYLE_LABELS, STYLE_PROMPTS, GeminiService
from .media import PDF_MIME, Media, prepare_image
from .ratelimit import RateLimiter
from .storage import Storage
from .utils import TELEGRAM_LIMIT, md_to_html, split_text

logger = logging.getLogger(__name__)

BTN_PHOTO = "📸 Как решить по фото"
BTN_MODE = "⚙️ Режим ответа"
BTN_NEW = "🔄 Новый диалог"
BTN_HELP = "❓ Помощь"

ALBUM_WAIT_SECONDS = 1.5  # столько ждём остальные фото альбома
MAX_MEDIA_PER_REQUEST = 10
MAX_TOTAL_MEDIA_MB = 18  # лимит Gemini на размер запроса с вложениями — около 20 МБ

GENERIC_ERROR = "Не удалось получить ответ. Попробуйте ещё раз чуть позже."

MAIN_KEYBOARD = ReplyKeyboardMarkup(
    [[BTN_PHOTO, BTN_MODE], [BTN_NEW, BTN_HELP]],
    resize_keyboard=True,
    input_field_placeholder="Задайте вопрос или отправьте фото задачи…",
)

START_TEXT = (
    "<b>Привет, {name}! 👋</b>\n\n"
    "Я — ИИ-ассистент на базе Google Gemini.\n\n"
    "📸 <b>Отправьте фото</b> примера, уравнения, задачи, учебника, теста или скриншота — "
    "я распознаю условие, решу и объясню решение.\n"
    "💬 <b>Задайте любой вопрос</b>: математика, физика, химия, история, программирование, "
    "тексты, переводы и многое другое.\n"
    "🧠 Я помню контекст разговора. Чтобы начать с чистого листа, нажмите «Новый диалог».\n\n"
    "Команды: /help — справка, /mode — стиль ответов, /reset — очистить историю."
)

HELP_TEXT = (
    "<b>Что я умею</b>\n"
    "• Решаю задачи по фото: математика, физика, химия, тесты, задания из тетради и учебника.\n"
    "• Отвечаю на вопросы по любым предметам и темам.\n"
    "• Пишу и исправляю тексты, сочинения, перевожу, объясняю сложное простыми словами.\n"
    "• Распознаю текст, графики, таблицы, схемы, документы и скриншоты.\n"
    "• Читаю PDF-файлы.\n"
    "• Помню контекст диалога — можно уточнять: «объясни шаг 3 подробнее».\n\n"
    "<b>Команды</b>\n"
    "/start — приветствие и меню\n"
    "/help — эта справка\n"
    "/mode — подробные или краткие ответы\n"
    "/reset — очистить историю диалога\n\n"
    "<b>Ограничения</b>\n"
    "• Фото из старых сообщений я не вижу — в памяти остаётся только текст диалога.\n"
    "• Ответы ИИ могут содержать ошибки: важные результаты проверяйте."
)

PHOTO_TIP_TEXT = (
    "<b>📸 Как получить лучший результат</b>\n"
    "1. Снимайте при хорошем освещении, ровно, чтобы вся задача была в кадре.\n"
    "2. Для мелкого текста отправляйте снимок <b>как файл</b> (без сжатия) — так распознавание точнее.\n"
    "3. Можно отправить несколько фото одним альбомом — я разберу их вместе.\n"
    "4. Подпись к фото — это ваша инструкция. Например: «Реши только №3» или «Объясни проще».\n"
    "5. Без подписи я просто решу задачу и подробно объясню решение.\n\n"
    "Попробуйте — отправьте фото прямо сейчас!"
)


def _seconds(value: Any) -> float:
    if isinstance(value, timedelta):
        return value.total_seconds()
    return float(value)


async def _reply(
    message: Message,
    text: str,
    *,
    html_mode: bool = False,
    reply_markup: Any = None,
) -> Message:
    """Отправка с повторами при флуд-лимите и сетевых сбоях Telegram.

    BadRequest (например, невалидный HTML) пробрасывается вызывающему коду.
    """
    kwargs: dict[str, Any] = {"reply_markup": reply_markup}
    if html_mode:
        kwargs["parse_mode"] = ParseMode.HTML
        kwargs["link_preview_options"] = LinkPreviewOptions(is_disabled=True)

    for attempt in range(1, 4):
        try:
            return await message.reply_text(text, **kwargs)
        except RetryAfter as exc:
            await asyncio.sleep(_seconds(exc.retry_after) + 1)
        except BadRequest:
            raise
        except (TimedOut, NetworkError):
            if attempt == 3:
                raise
            await asyncio.sleep(1.5 * attempt)
    raise RuntimeError("unreachable")  # pragma: no cover


async def _send_answer(message: Message, answer: str) -> None:
    """Отправляет длинный ответ частями; при ошибке разметки — обычным текстом."""
    for chunk in split_text(answer):
        rendered = md_to_html(chunk)
        try:
            if len(rendered) > TELEGRAM_LIMIT:
                raise BadRequest("rendered chunk is too long")
            await _reply(message, rendered, html_mode=True)
        except BadRequest as exc:
            logger.info("HTML-разметка отклонена (%s), отправляю простым текстом", exc)
            for part in split_text(chunk, 4000):
                await _reply(message, part)


class Typing:
    """Показывает «печатает…» всё время, пока выполняется блок."""

    def __init__(self, bot: Bot, chat_id: int) -> None:
        self._bot = bot
        self._chat_id = chat_id
        self._task: asyncio.Task[None] | None = None

    async def _run(self) -> None:
        while True:
            with contextlib.suppress(Exception):
                await self._bot.send_chat_action(self._chat_id, ChatAction.TYPING)
            await asyncio.sleep(4.5)

    async def __aenter__(self) -> "Typing":
        self._task = asyncio.create_task(self._run())
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        if self._task:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task


def _default_prompt(media: list[Media]) -> str:
    if any(item.is_pdf for item in media):
        return (
            "Проанализируй присланный документ: кратко опиши его содержание. "
            "Если в нём есть задания или вопросы — реши их и объясни решение."
        )
    target = "на изображении" if len(media) == 1 else "на этих изображениях"
    return (
        f"Определи, что {target}. Если это задача, пример, уравнение, тест или задание — "
        "реши и подробно объясни решение. Если это текст, график, таблица, схема или документ — "
        "распознай содержимое и кратко объясни его."
    )


def _history_label(media: list[Media]) -> str:
    images = sum(1 for item in media if not item.is_pdf)
    pdfs = len(media) - images
    parts = []
    if images:
        parts.append(f"изображений: {images}")
    if pdfs:
        parts.append(f"PDF: {pdfs}")
    return "[Пользователь прикрепил " + ", ".join(parts) + "]"


class BotHandlers:
    def __init__(
        self,
        settings: Settings,
        storage: Storage,
        gemini: GeminiService,
        limiter: RateLimiter,
    ) -> None:
        self._s = settings
        self._storage = storage
        self._gemini = gemini
        self._limiter = limiter
        self._locks: defaultdict[int, asyncio.Lock] = defaultdict(asyncio.Lock)
        self._albums: dict[tuple[int, str], list[Message]] = {}

    # ------------------------------------------------------------ registration
    def register(self, app: Application) -> None:
        private = filters.ChatType.PRIVATE
        new_message = private & filters.UpdateType.MESSAGE

        app.add_handler(CommandHandler("start", self.start, filters=private))
        app.add_handler(CommandHandler("help", self.help, filters=private))
        app.add_handler(CommandHandler("reset", self.reset, filters=private))
        app.add_handler(CommandHandler("mode", self.mode, filters=private))

        app.add_handler(MessageHandler(new_message & filters.Text([BTN_NEW]), self.reset))
        app.add_handler(MessageHandler(new_message & filters.Text([BTN_HELP]), self.help))
        app.add_handler(MessageHandler(new_message & filters.Text([BTN_MODE]), self.mode))
        app.add_handler(MessageHandler(new_message & filters.Text([BTN_PHOTO]), self.photo_tip))

        media_filter = filters.PHOTO | filters.Document.IMAGE | filters.Document.PDF
        app.add_handler(MessageHandler(new_message & media_filter, self.on_media))
        app.add_handler(MessageHandler(new_message & filters.TEXT & ~filters.COMMAND, self.on_text))
        app.add_handler(MessageHandler(new_message & ~filters.COMMAND, self.on_unsupported))

        app.add_handler(CallbackQueryHandler(self.on_mode_callback, pattern=r"^style:"))
        app.add_error_handler(self.on_error)

    async def post_init(self, app: Application) -> None:
        await app.bot.set_my_commands(
            [
                BotCommand("start", "Начать работу"),
                BotCommand("help", "Справка"),
                BotCommand("mode", "Подробные или краткие ответы"),
                BotCommand("reset", "Очистить историю диалога"),
            ]
        )
        me = await app.bot.get_me()
        logger.info("Бот запущен: @%s", me.username)
        await self._gemini.ping()

    async def post_shutdown(self, app: Application) -> None:
        self._storage.close()

    # ------------------------------------------------------------------ guards
    def _is_allowed(self, user_id: int) -> bool:
        allowed = self._s.allowed_user_ids
        return not allowed or user_id in allowed

    async def _check_access(self, message: Message, user: User) -> bool:
        if self._is_allowed(user.id):
            return True
        await _reply(message, "⛔ У вас нет доступа к этому боту.")
        return False

    async def _guard(self, message: Message, user: User) -> bool:
        """Доступ + защита от злоупотреблений (частота и суточный лимит)."""
        if not await self._check_access(message, user):
            return False

        wait = self._limiter.check(user.id)
        if wait > 0:
            await _reply(message, f"⏳ Слишком много запросов. Подождите {math.ceil(wait)} с.")
            return False

        limit = self._s.daily_limit_per_user
        if limit > 0 and not await self._storage.hit_daily(user.id, limit):
            await _reply(message, f"📅 Суточный лимит ({limit} запросов) исчерпан. Попробуйте завтра.")
            return False
        return True

    # ---------------------------------------------------------------- commands
    async def start(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        message, user = update.effective_message, update.effective_user
        if message is None or user is None or not await self._check_access(message, user):
            return
        name = html.escape(user.first_name or "друг")
        await _reply(
            message, START_TEXT.format(name=name), html_mode=True, reply_markup=MAIN_KEYBOARD
        )

    async def help(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        message, user = update.effective_message, update.effective_user
        if message is None or user is None or not await self._check_access(message, user):
            return
        await _reply(message, HELP_TEXT, html_mode=True, reply_markup=MAIN_KEYBOARD)

    async def photo_tip(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        message, user = update.effective_message, update.effective_user
        if message is None or user is None or not await self._check_access(message, user):
            return
        await _reply(message, PHOTO_TIP_TEXT, html_mode=True)

    async def reset(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        message, user = update.effective_message, update.effective_user
        if message is None or user is None or not await self._check_access(message, user):
            return
        async with self._locks[user.id]:
            await self._storage.clear_history(user.id)
        await _reply(message, "🧹 История диалога очищена. Начнём с чистого листа!")

    async def mode(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        message, user = update.effective_message, update.effective_user
        if message is None or user is None or not await self._check_access(message, user):
            return
        current = STYLE_LABELS.get(await self._storage.get_style(user.id), "")
        keyboard = InlineKeyboardMarkup(
            [
                [
                    InlineKeyboardButton(STYLE_LABELS["detailed"], callback_data="style:detailed"),
                    InlineKeyboardButton(STYLE_LABELS["brief"], callback_data="style:brief"),
                ]
            ]
        )
        await _reply(
            message,
            "⚙️ <b>Режим ответа</b>\n"
            f"Сейчас: {current}\n\n"
            "📖 <b>Подробно</b> — пошаговое решение с объяснением каждого шага.\n"
            "⚡ <b>Кратко</b> — только ответ и самое необходимое пояснение.",
            html_mode=True,
            reply_markup=keyboard,
        )

    async def on_mode_callback(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        query = update.callback_query
        if query is None or query.data is None:
            return
        if not self._is_allowed(query.from_user.id):
            await query.answer("Нет доступа", show_alert=True)
            return

        style = query.data.split(":", 1)[1]
        if style not in STYLE_PROMPTS:
            await query.answer("Неизвестный режим", show_alert=True)
            return

        await self._storage.set_style(query.from_user.id, style)
        await query.answer("Готово")
        with contextlib.suppress(BadRequest):
            await query.edit_message_text(f"✅ Режим ответа: {STYLE_LABELS[style]}")

    # ---------------------------------------------------------------- messages
    async def on_text(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        message, user = update.effective_message, update.effective_user
        if message is None or user is None or not message.text:
            return
        if not await self._guard(message, user):
            return

        prompt = message.text.strip()
        answer: str | None = None
        error_text = GENERIC_ERROR
        async with Typing(context.bot, message.chat_id):
            try:
                answer = await self._ask(user.id, prompt, [])
            except BotError as exc:
                error_text = exc.user_message
            except Exception:  # noqa: BLE001
                logger.exception("Ошибка при обработке текстового запроса")

        if answer is None:
            await _reply(message, f"⚠️ {error_text}")
            return
        await _send_answer(message, answer)

    async def on_media(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        message = update.effective_message
        if message is None or update.effective_user is None:
            return

        group_id = message.media_group_id
        if not group_id:
            await self._handle_media([message], context.bot)
            return

        # Альбом: Telegram присылает каждое фото отдельным обновлением.
        # Собираем их и обрабатываем одним запросом.
        key = (message.chat_id, group_id)
        bucket = self._albums.get(key)
        if bucket is not None:
            bucket.append(message)
            return
        self._albums[key] = [message]
        context.application.create_task(self._flush_album(key, context.bot), update=update)

    async def on_unsupported(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        message, user = update.effective_message, update.effective_user
        if message is None or user is None or not await self._check_access(message, user):
            return
        await _reply(
            message,
            "Я работаю с текстом, фотографиями, изображениями-файлами и PDF. "
            "Пришлите вопрос или фото задачи 🙂",
        )

    # ------------------------------------------------------------------- media
    async def _flush_album(self, key: tuple[int, str], bot: Bot) -> None:
        await asyncio.sleep(ALBUM_WAIT_SECONDS)
        messages = self._albums.pop(key, [])
        if messages:
            messages.sort(key=lambda m: m.message_id)
            await self._handle_media(messages, bot)

    async def _handle_media(self, messages: list[Message], bot: Bot) -> None:
        first = messages[0]
        user = first.from_user
        if user is None or not await self._guard(first, user):
            return

        caption = next((m.caption.strip() for m in messages if m.caption and m.caption.strip()), "")
        status = await _reply(first, "🔍 Анализирую… это может занять до минуты.")

        answer: str | None = None
        error_text = GENERIC_ERROR
        try:
            async with Typing(bot, first.chat_id):
                media = await self._collect_media(messages, bot)
                prompt = caption or _default_prompt(media)
                answer = await self._ask(user.id, prompt, media)
        except BotError as exc:
            error_text = exc.user_message
        except Exception:  # noqa: BLE001
            logger.exception("Ошибка при обработке файла")
        finally:
            with contextlib.suppress(Exception):
                await status.delete()

        if answer is None:
            await _reply(first, f"⚠️ {error_text}")
            return
        await _send_answer(first, answer)

    async def _collect_media(self, messages: list[Message], bot: Bot) -> list[Media]:
        max_bytes = self._s.max_file_mb * 1024 * 1024
        total_limit = MAX_TOTAL_MEDIA_MB * 1024 * 1024
        too_big = MediaError(
            f"Файл слишком большой. Максимальный размер — {self._s.max_file_mb} МБ."
        )

        items: list[Media] = []
        total = 0
        for message in messages[:MAX_MEDIA_PER_REQUEST]:
            if message.photo:
                tg_obj = message.photo[-1]  # самое большое разрешение
                is_pdf = False
            elif message.document:
                tg_obj = message.document
                is_pdf = (message.document.mime_type or "").lower() == PDF_MIME
            else:
                continue

            if tg_obj.file_size and tg_obj.file_size > max_bytes:
                raise too_big

            try:
                tg_file = await bot.get_file(tg_obj.file_id)
                raw = bytes(await tg_file.download_as_bytearray())
            except BadRequest as exc:
                if "too big" in str(exc).lower():
                    raise MediaError("Telegram не разрешает ботам скачивать файлы больше 20 МБ.") from exc
                raise

            if len(raw) > max_bytes:
                raise too_big

            if is_pdf:
                item = Media(data=raw, mime_type=PDF_MIME)
            else:
                jpeg = await asyncio.to_thread(prepare_image, raw)
                item = Media(data=jpeg, mime_type="image/jpeg")

            total += len(item.data)
            if total > total_limit:
                raise MediaError(
                    f"Суммарный размер файлов слишком велик (максимум {MAX_TOTAL_MEDIA_MB} МБ). "
                    "Отправьте меньше файлов за раз."
                )
            items.append(item)

        if not items:
            raise MediaError("Не удалось найти в сообщении поддерживаемый файл.")
        return items

    # --------------------------------------------------------------------- core
    async def _ask(self, user_id: int, prompt: str, media: list[Media]) -> str:
        """Запрос к Gemini с учётом истории. Запросы одного пользователя идут по очереди."""
        async with self._locks[user_id]:
            history = await self._storage.get_history(user_id)
            style = await self._storage.get_style(user_id)
            answer = await self._gemini.generate(history, prompt, media, style)

            stored_prompt = f"{_history_label(media)} {prompt}" if media else prompt
            await self._storage.add_exchange(user_id, stored_prompt, answer)
        return answer

    # ------------------------------------------------------------------- errors
    async def on_error(self, update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
        error = context.error
        if isinstance(error, (TimedOut, NetworkError)) and not isinstance(error, BadRequest):
            logger.warning("Сетевая ошибка Telegram: %s", error)
            return

        logger.error("Необработанная ошибка", exc_info=error)
        if isinstance(update, Update) and update.effective_message is not None:
            with contextlib.suppress(Exception):
                await update.effective_message.reply_text(
                    "⚠️ Произошла внутренняя ошибка. Попробуйте ещё раз чуть позже."
                )

"""Форматирование ответов: разбиение длинных сообщений и Markdown → Telegram HTML."""
from __future__ import annotations

import html
import re

TELEGRAM_LIMIT = 4096  # максимум символов в одном сообщении Telegram
SPLIT_LIMIT = 3500  # запас под HTML-теги и экранирование


def _hard_wrap(line: str, width: int) -> list[str]:
    """Режет слишком длинную строку, по возможности по пробелам."""
    if len(line) <= width:
        return [line]
    parts: list[str] = []
    while len(line) > width:
        cut = line.rfind(" ", 0, width)
        if cut < width // 2:
            cut = width
        parts.append(line[:cut])
        line = line[cut:].lstrip(" ")
    parts.append(line)
    return parts


def split_text(text: str, limit: int = SPLIT_LIMIT) -> list[str]:
    """Делит текст на части не длиннее limit, не разрывая блоки кода ``` ... ```.

    Если разрыв приходится внутрь блока кода, блок закрывается в конце части
    и открывается заново в начале следующей.
    """
    text = text.strip()
    if not text:
        return []
    if len(text) <= limit:
        return [text]

    chunks: list[str] = []
    current: list[str] = []
    size = 0
    in_code = False
    lang = ""

    def flush(reopen: bool) -> None:
        nonlocal current, size
        body = "\n".join(current)
        if in_code:
            body += "\n```"
        body = body.strip("\n")
        if body.strip():
            chunks.append(body)
        current, size = [], 0
        if reopen and in_code:
            header = f"```{lang}"
            current.append(header)
            size = len(header) + 1

    for raw_line in text.split("\n"):
        for line in _hard_wrap(raw_line, limit - 20):
            reserve = 4 if in_code else 0
            if current and size + len(line) + 1 + reserve > limit:
                flush(reopen=True)
            current.append(line)
            size += len(line) + 1
            stripped = line.strip()
            if stripped.startswith("```"):
                if in_code:
                    in_code = False
                    lang = ""
                else:
                    in_code = True
                    lang = stripped[3:].strip()

    if current:
        flush(reopen=False)
    return chunks


_FENCE_RE = re.compile(r"```[ \t]*([\w+#.\-]*)[^\n]*\n(.*?)(?:```|\Z)", re.DOTALL)
_TABLE_RE = re.compile(r"(?:^[ \t]*\|.*\|[ \t]*(?:\n|\Z)){2,}", re.MULTILINE)
_INLINE_CODE_RE = re.compile(r"`([^`\n]+)`")


def md_to_html(text: str) -> str:
    """Преобразует Markdown, который выдаёт Gemini, в HTML-подмножество Telegram.

    Поддерживается: **жирный**, *курсив*, ~~зачёркнутый~~, `код`, блоки кода, заголовки,
    маркированные списки, ссылки. Таблицы выводятся моноширинным блоком.
    Если Telegram отклонит разметку, обработчик отправит текст без форматирования.
    """
    stash: list[str] = []

    def keep(fragment: str) -> str:
        stash.append(fragment)
        return f"\x00{len(stash) - 1}\x00"

    def fence(match: re.Match[str]) -> str:
        lang = match.group(1)
        code = html.escape(match.group(2).rstrip("\n"), quote=False)
        if lang:
            return keep(f'<pre><code class="language-{html.escape(lang)}">{code}</code></pre>')
        return keep(f"<pre>{code}</pre>")

    def table(match: re.Match[str]) -> str:
        return keep(f"<pre>{html.escape(match.group(0).strip(chr(10)), quote=False)}</pre>")

    text = _FENCE_RE.sub(fence, text)
    text = _TABLE_RE.sub(table, text)
    text = _INLINE_CODE_RE.sub(
        lambda m: keep(f"<code>{html.escape(m.group(1), quote=False)}</code>"), text
    )
    text = html.escape(text, quote=False)

    # Заголовки → жирный текст
    text = re.sub(
        r"^[ \t]*#{1,6}[ \t]+(.+?)[ \t#]*$",
        lambda m: "<b>" + m.group(1).replace("**", "") + "</b>",
        text,
        flags=re.MULTILINE,
    )
    # Горизонтальные линии
    text = re.sub(r"^[ \t]*([-*_])(?:[ \t]*\1){2,}[ \t]*$", "──────────", text, flags=re.MULTILINE)
    # Маркеры списков
    text = re.sub(r"^([ \t]*)[*+\-][ \t]+", r"\1• ", text, flags=re.MULTILINE)
    # Жирный, курсив, зачёркнутый
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"(?<![\w_])__(.+?)__(?![\w_])", r"<b>\1</b>", text)
    text = re.sub(r"(?<![\w*])\*(?![\s*])(.+?)(?<![\s*])\*(?![\w*])", r"<i>\1</i>", text)
    text = re.sub(r"~~(.+?)~~", r"<s>\1</s>", text)
    # Ссылки
    text = re.sub(
        r"\[([^\]\n]+)\]\((https?://[^\s)\"]+)\)", r'<a href="\2">\1</a>', text
    )

    text = re.sub(r"\x00(\d+)\x00", lambda m: stash[int(m.group(1))], text)
    return text.strip()

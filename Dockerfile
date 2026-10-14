FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Непривилегированный пользователь
RUN useradd --create-home --uid 1000 bot

COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY bot ./bot
COPY scripts ./scripts

# Каталог данных (SQLite) принадлежит пользователю bot; поверх него монтируется том
RUN mkdir -p /app/data && chown -R bot:bot /app

USER bot

CMD ["python", "-m", "bot.main"]

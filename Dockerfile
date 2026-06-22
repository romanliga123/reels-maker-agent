FROM python:3.11-slim

# ffmpeg — нужен пайплайну напрямую; libsm6/libxext6/libxrender1/libglib2.0-0 —
# транзитивные зависимости cv2 (opencv-python-headless) при импорте на Debian slim.
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg libsm6 libxext6 libxrender1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PYTHONUNBUFFERED=1

# Сессии и JobLoop живут в памяти процесса — обязательно один воркер,
# иначе запросы одной сессии могут попасть на разные процессы.
CMD ["sh", "-c", "uvicorn web.server:app --host 0.0.0.0 --port ${PORT:-8010} --workers 1"]

FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# yfinance/akshare는 requests 계열만 사용 — 의존성 최소화
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt gunicorn

COPY app.py run_server.sh ./
COPY core/ ./core/
COPY static/ ./static/

EXPOSE 8787

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8787/api/health', timeout=4)"

# 단일 워커 + SSE 스트리밍용 타임아웃 여유
CMD ["gunicorn", "app:app", \
     "-k", "uvicorn.workers.UvicornWorker", \
     "-b", "0.0.0.0:8787", \
     "-w", "2", \
     "--timeout", "180", \
     "--graceful-timeout", "30", \
     "--access-logfile", "-"]

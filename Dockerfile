FROM python:3.11-slim
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*
COPY pyproject.toml README.md ./
RUN python -c "import pathlib,tomllib; p=tomllib.loads(pathlib.Path('pyproject.toml').read_text()); pathlib.Path('/tmp/requirements.txt').write_text('\n'.join(p['project']['dependencies']+p['project']['optional-dependencies']['app']))" && pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch==2.13.0 torchvision==0.28.0 && pip install --no-cache-dir -r /tmp/requirements.txt
COPY src ./src
RUN pip install --no-cache-dir --no-deps .
COPY processing/roi_config.json ./processing/roi_config.json
COPY migrations ./migrations
COPY alembic.ini ./
ENV YOLO_CONFIG_DIR=/home/inspector/.config/Ultralytics
ENV INSPECTOR_ROOT=/app
RUN useradd --uid 10001 --create-home inspector && mkdir -p /app/var/inspections /home/inspector/.config/Ultralytics && chown -R inspector:inspector /app/var /home/inspector
USER inspector
EXPOSE 8501
CMD ["sh", "-c", "python -m alembic upgrade head && exec python -m uvicorn glass_inspector.server:app --host 0.0.0.0 --port 8501 --workers 1"]

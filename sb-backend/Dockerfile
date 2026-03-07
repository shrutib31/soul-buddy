FROM python:3.11-slim

WORKDIR /app

# Copy requirements and install dependencies first (layer-cached separately from source)
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the backend source
COPY . .

RUN chmod +x /app/entrypoint.sh

# Pre-create log directory so it exists before setup_logging runs
RUN mkdir -p /app/logs

# Default environment variables (overrideable at runtime via env_file / -e)
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=8000

EXPOSE 8000

ENTRYPOINT ["/app/entrypoint.sh"]

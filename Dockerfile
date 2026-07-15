FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd -m -u 1000 appuser && \
    mkdir -p /app/media/uploads /app/media/gradcam /app/media/preprocessing_viz \
             /app/models /app/logs /app/experiments && \
    chown -R appuser:appuser /app

# Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Application code
COPY . .

# Run migrations and collect static
RUN python manage.py migrate --noinput 2>/dev/null || true
RUN python manage.py collectstatic --noinput 2>/dev/null || true

# Set ownership
RUN chown -R appuser:appuser /app

USER appuser

EXPOSE 8000

# Use gunicorn config file
CMD ["gunicorn", "-c", "gunicorn.conf.py", "retina_project.wsgi:application"]

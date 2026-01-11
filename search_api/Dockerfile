# Dockerfile for Ad Search API (always-running service)
FROM python:3.11-slim

WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY search_api.py .

# Expose port (Railway will set PORT dynamically)
EXPOSE 8000

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Run the API
# Railway sets PORT environment variable dynamically
# Default to 8000 if PORT is not set
CMD uvicorn search_api:app --host 0.0.0.0 --port ${PORT:-8000}

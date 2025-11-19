FROM python:3.10-slim

WORKDIR /app

# Install system dependencies for OpenCV
RUN apt-get update && apt-get install -y \
    libsm6 libxext6 libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy backend requirements
COPY Backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app
COPY . .

EXPOSE 7860

# Health check
HEALTHCHECK CMD python -c "import requests; requests.get('http://localhost:7860/docs')" || exit 1

# Run FastAPI backend
CMD ["python", "main.py"]

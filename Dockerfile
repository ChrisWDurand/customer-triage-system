# Use a slim Python base image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy files into the container
COPY . /app

# Install dependencies
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Expose port for FastAPI
EXPOSE 8000

# Default command: start the API server
CMD ["uvicorn", "serve.api:app", "--host", "0.0.0.0", "--port", "8000"]
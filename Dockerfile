FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --timeout=120 -r requirements.txt

# Copy application code
COPY . .

# Build the vector index at image build time (mock mode)
ENV USE_MOCK_LLM=true
RUN python -m scripts.build_index

# Expose the application port
EXPOSE 8000

# Run the FastAPI server
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

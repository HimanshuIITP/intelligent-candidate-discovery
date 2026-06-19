FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ ./src/
COPY rank.py setup_models.py ./

# Download model during build (so ranking has no network dependency)
RUN python setup_models.py

# Copy data (mounted at runtime in practice)
COPY data/ ./data/

# Default entrypoint
ENTRYPOINT ["python", "rank.py"]
CMD ["--candidates", "./data/candidates.jsonl", "--out", "./submission.csv"]

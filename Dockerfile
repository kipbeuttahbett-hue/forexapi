FROM python:3.13-slim

# Set working directory
WORKDIR /app

# Install system dependencies (often needed for pandas/numpy)
RUN apt-get update && apt-get install -y build-essential && rm -rf /var/lib/apt/lists/*

# Install python packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code and model files
COPY api.py .
COPY gold_decision_tree.pkl .
COPY gold_features.pkl .

# Expose the port
EXPOSE 5000

# Run the application
CMD ["python", "api.py"]

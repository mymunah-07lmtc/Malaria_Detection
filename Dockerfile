# Use Python 3.11 (which has wheels for tflite-runtime)
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the app
COPY . .

# Expose the port (Hugging Face Spaces uses 7860 by default)
EXPOSE 7860

# Run the app
CMD ["streamlit", "run", "app_pwa.py", "--server.port=7860", "--server.address=0.0.0.0"]
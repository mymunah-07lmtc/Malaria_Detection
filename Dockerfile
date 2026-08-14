FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 10000

CMD ["streamlit", "run", "app_pwa.py", "--server.port=10000", "--server.address=0.0.0.0", "--server.enableCORS=false", "--server.enableXsrfProtection=false"]
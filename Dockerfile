   FROM python:3.11-slim

   ENV PYTHONDONTWRITEBYTECODE=1 \
       PYTHONUNBUFFERED=1

   WORKDIR /app

   COPY requirements.txt .
   RUN pip install --no-cache-dir -r requirements.txt

   COPY . .

   EXPOSE 8501

   CMD ["sh", "-c", "mkdir -p /app/.streamlit && cp /etc/secrets/secrets.toml /app/.streamlit/secrets.toml 2>/dev/null; streamlit run app.py --server.port=${PORT:-8501} --server.address=0.0.0.0 --server.headless=true"]

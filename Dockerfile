FROM python:3.12-slim

WORKDIR /app
RUN chmod 777 /tmp
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8080

CMD ["gunicorn", "-b", "0.0.0.0:8080", "app:server"]

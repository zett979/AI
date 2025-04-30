FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt /app/
RUN pip install --retries 5 --no-cache-dir -r requirements.txt

COPY . /app/

EXPOSE 8050

CMD ["gunicorn", "-b", "0.0.0.0:8050", "--reload", "app:server"]

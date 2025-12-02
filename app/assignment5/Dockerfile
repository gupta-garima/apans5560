FROM python:3.11-slim

WORKDIR /app

COPY rl_trained_gpt2 /app/rl_trained_gpt2
COPY main.py /app/main.py
COPY requirements.txt /app/requirements.txt

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]


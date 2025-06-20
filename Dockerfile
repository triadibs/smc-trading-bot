FROM python:3.10-slim

WORKDIR /app

COPY main.py /app/

# Install dependencies jika ada, misal:
COPY requirements.txt /app/
 RUN pip install -r requirements.txt

CMD ["python", "main.py"]
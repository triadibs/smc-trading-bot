FROM python:3.11-slim
WORKDIR /app
COPY main.py .
RUN pip install ccxt nest_asyncio python-telegram-bot pandas google-generativeai
CMD ["python", "main.py"]

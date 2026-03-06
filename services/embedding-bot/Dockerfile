FROM python:3.11-slim
WORKDIR /app
COPY ./shared /app/shared
COPY ./services/embedding-bot/requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt
COPY ./services/embedding-bot /app/
EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
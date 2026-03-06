# Use a lightweight Python 3.11 image
FROM python:3.11-slim

# Set the working directory inside the container
WORKDIR /app

# 1. Copy the shared directory first so the bot can import shared.models
COPY ./shared /app/shared

# 2. Copy the specific bot's requirements and install them
COPY ./graph-bot/requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# 3. Copy the rest of the bot's code
COPY ./graph-bot /app/

# Expose the port Uvicorn will run on
EXPOSE 8000

# Start the FastAPI server
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
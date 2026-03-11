FROM python:3.11-slim

WORKDIR /app

# Install Streamlit and Requests (to talk to the Orchestrator)
RUN pip install streamlit requests

COPY . /app

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
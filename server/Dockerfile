FROM python:3.10-slim

WORKDIR /server
COPY server/requirements.txt .
RUN ["pip", "install", "-r", "requirements.txt"]

COPY server/server.py .

CMD ["python", "server.py"]
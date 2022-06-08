FROM python:3.10-slim

WORKDIR /client
COPY client/requirements.txt .
RUN ["pip", "install", "-r", "requirements.txt"]

COPY client/client.py .

CMD  waitress-serve --port=8080 client:server
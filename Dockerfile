FROM tiangolo/uvicorn-gunicorn-fastapi:python3.9

COPY ./models /models

COPY ./app /app

COPY requirements.txt requirements.txt

RUN pip install -r requirements.txt
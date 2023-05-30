FROM tiangolo/uvicorn-gunicorn-fastapi:python3.9

COPY ./app /app

COPY ./test-pythia-70m /app/test-pythia-70m

COPY requirements.txt requirements.txt

RUN pip install -r requirements.txt
#This article show how to push and pull to GAR https://cloud.google.com/artifact-registry/docs/docker/pushing-and-pulling
FROM python:3.10-slim-buster

# 
WORKDIR /code
 
# 
COPY ./requirements.txt /code/requirements.txt

# 
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# 
COPY ./app /app

# 
CMD ["uvicorn", "main:app", "--proxy-headers", "--host", "0.0.0.0", "--port", "80"]
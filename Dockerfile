FROM tiangolo/uvicorn-gunicorn-fastapi:python3.9

COPY ./requirements.txt /app/requirements.txt

RUN apt-get update \
  && apt-get -y install tesseract-ocr
  
RUN pip install --no-cache-dir --upgrade -r /app/requirements.txt

COPY ./app /app/app

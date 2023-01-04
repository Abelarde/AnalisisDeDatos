FROM python:3.10-slim

WORKDIR /code

ENV FLASK_APP app.py

ENV FLASK_RUN_HOST 0.0.0.0

COPY requirements.txt requirements.txt

RUN pip install -r requirements.txt --no-cache-dir

COPY . /code

CMD ["flask", "run"]
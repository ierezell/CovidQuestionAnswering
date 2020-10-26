FROM python:3.8

WORKDIR /usr/src/app

COPY . .

RUN pip install --no-cache-dir -r requirements.txt

CMD [ "gunicorn", "run_results:server", "--preload"]
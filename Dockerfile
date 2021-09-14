FROM python:3.7

RUN apt-get update
RUN apt-get install -y nethogs
RUN apt-get install -y net-tools

COPY requirements.txt /app/requirements.txt
RUN pip install -r /app/requirements.txt

COPY . /app
RUN pip install /app

EXPOSE 6400

CMD python /app/examples/start_distributed_object_remote.py
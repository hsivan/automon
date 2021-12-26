FROM python:3.7

RUN apt-get update
RUN apt-get install -y nethogs
RUN apt-get install -y net-tools

RUN pip install boto3==1.18.21
RUN pip install botocore==1.21.65
RUN pip install matplotlib==3.4.2
RUN pip install pandas==1.3.2

COPY requirements.txt /app/requirements.txt
RUN pip install -r /app/requirements.txt


COPY . /app
RUN pip install /app

EXPOSE 6400
ENV PYTHONPATH "${PYTHONPATH}:/app"

CMD python /app/examples/start_distributed_object_remote.py
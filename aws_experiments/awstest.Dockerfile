FROM python:3.7

RUN apt-get update
RUN apt-get install -y nethogs
RUN apt-get install -y net-tools

RUN pip install boto3
RUN pip install botocore
RUN pip install paramiko
RUN pip install matplotlib
RUN pip install pandas

COPY requirements.txt /app/requirements.txt
RUN pip install -r /app/requirements.txt

COPY . /app
RUN pip install /app

EXPOSE 6400
ENV PYTHONPATH "${PYTHONPATH}:/app"

CMD python /app/aws_experiments/start_distributed_object_remote.py
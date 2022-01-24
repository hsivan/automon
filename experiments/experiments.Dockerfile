FROM python:3.7

RUN apt-get update
RUN apt-get install -y nethogs
RUN apt-get install -y net-tools

RUN pip install matplotlib
RUN pip install pandas

COPY requirements.txt /app/requirements.txt
RUN pip install -r /app/requirements.txt

COPY . /app
RUN pip install /app

EXPOSE 6400
ENV PYTHONPATH "${PYTHONPATH}:/app"

WORKDIR "/app/experiments"
CMD python test_max_error_vs_communication_inner_product.py
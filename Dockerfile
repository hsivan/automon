FROM python:3.7

COPY requirements.txt /app/requirements.txt
RUN pip install -r /app/requirements.txt

COPY . /app
RUN pip install /app

EXPOSE 6400
ENV PYTHONPATH "${PYTHONPATH}:/app"

CMD python /app/examples/simple_automon_coordinator.py
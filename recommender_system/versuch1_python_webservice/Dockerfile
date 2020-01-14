FROM python:3.7-slim-buster
WORKDIR /app
EXPOSE 8080

RUN pip3 install --upgrade pip
RUN pip3 install pandas==0.25.2
RUN pip3 install tensorflow==2.0.0
RUN pip3 install keras==2.3.1

RUN pip3 install ptvsd==4.3.2
RUN pip3 install Flask==1.1.1

ENV PYTHONUNBUFFERED=0

COPY ["tokenizer", "/app/tokenizer"]
COPY ["models", "/app/models"]
COPY ["predictionServer.py", "/app/predictionServer.py"]


ENTRYPOINT ["python", "predictionServer.py"]
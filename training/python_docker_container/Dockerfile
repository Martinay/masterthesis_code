FROM python:3.7-buster

VOLUME /files
WORKDIR /files

RUN pip3 install ptvsd==4.3.2 && \
    pip3 install pandas==0.25.2

RUN pip3 install tensorflow==2.0.0 && \
    pip3 install tensorflow-addons==0.6.0 && \
    pip3 install keras==2.3.1 && \
    pip3 install keras-tuner==1.0.0

RUN apt-get update  && \
    apt-get install graphviz -y && \
    pip3 install pydot==1.4.1

RUN pip3 install git+https://github.com/onnx/keras-onnx.git@tf20 && \
    pip3 install onnx==1.6.0
FROM nvcr.io/nvidia/pytorch:24.01-py3
ARG DEBIAN_FRONTEND=noninteractive
WORKDIR /openduck-py

COPY ./openduck-py/requirements.txt /openduck-py/requirements.txt

RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y uvicorn gunicorn awscli espeak-ng && \
    apt-get clean

RUN pip install --upgrade pip
RUN pip install Cython
RUN pip install --no-cache-dir -r requirements.txt

COPY ./openduck-py /openduck-py

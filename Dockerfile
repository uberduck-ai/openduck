FROM nvcr.io/nvidia/pytorch:24.01-py3
ARG DEBIAN_FRONTEND=noninteractive
WORKDIR /openduck-py

COPY ./openduck-py/requirements.txt /openduck-py/requirements.txt

RUN wget http://archive.ubuntu.com/ubuntu/pool/main/o/openssl/libssl1.1_1.1.0g-2ubuntu4_amd64.deb
RUN dpkg -i libssl1.1_1.1.0g-2ubuntu4_amd64.deb

RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y uvicorn gunicorn awscli espeak-ng && \
    apt-get clean

RUN pip install --upgrade pip
RUN pip install Cython
RUN pip install --no-cache-dir -r requirements.txt

COPY ./openduck-py /openduck-py

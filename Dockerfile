FROM python:3.11

WORKDIR /openduck-py

COPY openduck-py/requirements.txt /openduck-py/

RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y uvicorn awscli && \ 
    apt-get clean

RUN pip install --upgrade pip
RUN pip install Cython
RUN pip install --no-cache-dir -r requirements.txt
COPY ./openduck-py /openduck-py
WORKDIR /openduck-py
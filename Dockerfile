FROM nvcr.io/nvidia/pytorch:24.01-py3 AS builder
ARG DEBIAN_FRONTEND=noninteractive
WORKDIR /openduck-py

COPY ./openduck-py/requirements.txt /openduck-py/requirements.txt

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
    uvicorn \
    gunicorn \
    awscli \
    espeak-ng \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    && pip install --upgrade pip \
    && pip install Cython \
    && pip install --no-cache-dir -r requirements.txt

FROM nvcr.io/nvidia/pytorch:24.01-py3
ARG DEBIAN_FRONTEND=noninteractive
WORKDIR /openduck-py

COPY --from=builder /usr/local/lib/python3.8/site-packages/ /usr/local/lib/python3.8/site-packages/
COPY --from=builder /openduck-py /openduck-py
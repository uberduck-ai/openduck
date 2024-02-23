FROM python:3.11
# nvcr.io/nvidia/pytorch:xx.xx-py3


# ARG DEBIAN_FRONTEND=noninteractive

WORKDIR /openduck-py

COPY openduck-py/requirements.txt /openduck-py/

# Run commands inside the image
# Update and upgrade the package list, then install the Python dependencies
RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y uvicorn awscli && \ 
    apt-get clean

RUN pip install --upgrade pip
RUN pip install Cython
RUN pip install --no-cache-dir -r requirements.txt
COPY ./openduck-py /openduck-py
WORKDIR /openduck-py


# TODO: Copy in the prototype models from S3 instead of assuming the user has them on disk

# Run uvicorn when the container launches
# CMD ["uvicorn", "openduck_py.routers.main:app", "--reload", "--host", "0.0.0.0", "--port", "8000"]

FROM python:3.10-slim-bookworm

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    gcc \
    g++ \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip setuptools wheel

RUN pip install setuptools==68.2.2

RUN pip install numpy==1.23.5

COPY ./requirements.txt /opt/

RUN pip install --no-cache-dir --no-build-isolation -r /opt/requirements.txt

COPY ./dcm2nifti_processing.py /opt/
COPY ./radiomics_utils.py /opt/

ENV PYTHONUNBUFFERED=TRUE
ENV AWS_DEFAULT_REGION=eu-west-2
ENV PYTHONPATH="/opt"

WORKDIR /opt/

ENTRYPOINT ["python", "dcm2nifti_processing.py"]
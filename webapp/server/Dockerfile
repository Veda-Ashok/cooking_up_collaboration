
FROM python:3.10-slim

ARG BUILD_ENV
ARG OVERCOOKED_BRANCH
ARG HARL_BRANCH
ARG GRAPHICS

WORKDIR /app

RUN apt-get -y update && apt-get install -y \
    git \
    gcc \
    g++ \
    gfortran \
    libopenblas-dev \
    liblapack-dev \
    libgl1-mesa-dev \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip first
RUN pip install --upgrade pip setuptools wheel

# Install numpy separately first to avoid build issues
RUN pip install "numpy<2.0"

COPY ./requirements.txt ./requirements.txt
RUN pip install -r requirements.txt

# Install eventlet production server if production build
RUN if [ "$BUILD_ENV" = "production" ] ; then pip install eventlet ; fi


RUN git clone https://github.com/adityakotha03/overcooked_ai.git /overcooked_ai && \
    cd /overcooked_ai && \
    pip install -e .


COPY ./static ./static
COPY ./*.py ./
COPY ./graphics/$GRAPHICS ./static/js/graphics.js
COPY ./config.json ./config.json

# Set environment variables that will be used by app.py
ENV HOST 0.0.0.0
ENV PORT 5000
ENV CONF_PATH config.json
ENV PYTHONPATH=/overcooked_ai:$PYTHONPATH

# Do the thing
EXPOSE 5000
CMD ["python", "-u", "app.py"]

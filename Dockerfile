ARG PYTHON_VERSION=3.10
FROM python:${PYTHON_VERSION}-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Create experiment directory
RUN mkdir -p /experiment

# Set up volume for data
VOLUME "/data"
ENV DATA_DIR=/data

# Set working directory
WORKDIR /experiment

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install dependencies with pip upgrade and extra index url
RUN python -m pip install --no-cache-dir --upgrade pip && \
    pip install --default-timeout=3000 --no-cache-dir -r requirements.txt


# Copy all experiment files
COPY clip.py /experiment/

# Set owner for output files
ENV OWNER=1000:1000

# Run experiment
CMD export OUTPUT_DIR=$DATA_DIR/$(date +%Y-%m-%d-%H-%M-%S)-$(hostname) && \
    mkdir -p $OUTPUT_DIR && \
    python clip.py | tee $OUTPUT_DIR/output.log && \
    chown -R $OWNER $DATA_DIR
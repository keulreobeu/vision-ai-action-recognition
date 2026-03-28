FROM python:3.12-slim

WORKDIR /workspace

RUN apt-get update && apt-get install -y \
    build-essential \
    ffmpeg \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

COPY docker/requirements.app.txt /tmp/requirements.app.txt
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r /tmp/requirements.app.txt

COPY 2_preprocessing /workspace/2_preprocessing
COPY 4_predict /workspace/4_predict
COPY 5_langchain /workspace/5_langchain

CMD ["bash"]
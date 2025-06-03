FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-setuptools \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip3 install --no-cache-dir --timeout=300 --retries=10 -r requirements.txt

COPY entrypoint.sh .
RUN chmod +x ./entrypoint.sh

COPY . .

ENTRYPOINT ["./entrypoint.sh"]
FROM python:3.9-slim

RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y\
    git \
    && rm -rf /var/lib/apt/lists/*


WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python", "predict.py"]

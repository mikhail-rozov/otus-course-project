FROM python:3.10

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt
RUN wget "https://storage.yandexcloud.net/cloud-certs/CA.pem"

COPY . .

ENTRYPOINT [ "python3", "model_inference.py" ]
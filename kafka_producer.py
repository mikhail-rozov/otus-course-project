from kafka import KafkaProducer


FILE_TO_SEND = "test.csv"
BOOTSTRAP_SERVERS = ['rc1a-ludjnvo4fc5b2k1r.mdb.yandexcloud.net:9091']


def send_to_kafka(producer, topic, chunk):
    for line in chunk:
        producer.send(topic, value=line.encode("utf-8"))
    producer.flush()


def process_file_in_chunks(file_path, producer, topic, chunk_size=50000):
    with open(file_path, 'r') as file:
        chunk = []
        for i, line in enumerate(file):
            chunk.append(line.strip())
            if (i + 1) % chunk_size == 0:
                send_to_kafka(producer, topic, chunk)
                chunk.clear()

        if chunk:
            send_to_kafka(producer, topic, chunk)


def main():
    producer = KafkaProducer(
        bootstrap_servers=BOOTSTRAP_SERVERS,
        security_protocol="SASL_SSL",
        sasl_mechanism="SCRAM-SHA-512",
        sasl_plain_username='mlops',
        sasl_plain_password='otus-mlops',
        ssl_cafile="/usr/local/share/ca-certificates/Yandex/YandexInternalRootCA.crt"
    )

    topic = 'input'
    process_file_in_chunks(FILE_TO_SEND, producer, topic)


if __name__ == "__main__":
    main()

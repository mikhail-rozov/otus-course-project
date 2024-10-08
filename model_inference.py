import pickle

from kafka import KafkaConsumer, KafkaProducer
import pandas as pd


KAFKA_SERVERS = ['rc1a-ludjnvo4fc5b2k1r.mdb.yandexcloud.net:9091']

def create_time_features(df: pd.DataFrame):

    df['datetime'] = pd.to_datetime(df['click_time'])
    df['day_of_week'] = df['datetime'].dt.dayofweek
    df['hour'] = df['datetime'].dt.hour
    df.drop("datetime", axis=1, inplace=True)
           
    return df


def create_grouped_features(df):

    ip_count = df.groupby('ip').size().reset_index(name='ip_count').astype(int)
    ip_day_hour = df.groupby(['ip','day_of_week','hour']).size().reset_index(name='ip_day_hour').astype(int)
    ip_hour_channel = df[['ip','hour','channel']].groupby(['ip','hour','channel']).size().reset_index(name='ip_hour_channel').astype(int)
    ip_hour_os = df.groupby(['ip', 'hour', 'os']).channel.count().reset_index(name='ip_hour_os').astype(int)
    ip_hour_app = df.groupby(['ip', 'hour', 'app']).channel.count().reset_index(name='ip_hour_app').astype(int)
    ip_hour_device = df.groupby(['ip', 'hour', 'device']).channel.count().reset_index(name='ip_hour_device').astype(int)

    df = pd.merge(df, ip_count, on='ip', how='left')
    del ip_count
    df = pd.merge(df, ip_day_hour, on=['ip', 'day_of_week', 'hour'], how='left')
    del ip_day_hour
    df = pd.merge(df, ip_hour_channel, on=['ip', 'hour', 'channel'], how='left')
    del ip_hour_channel
    df = pd.merge(df, ip_hour_os, on=['ip', 'hour', 'os'], how='left')
    del ip_hour_os
    df = pd.merge(df, ip_hour_app, on=['ip', 'hour', 'app'], how='left')
    del ip_hour_app
    df = pd.merge(df, ip_hour_device, on=['ip', 'hour', 'device'], how='left')
    del ip_hour_device

    return df


def main():

    consumer = KafkaConsumer(
        'input',
        bootstrap_servers=KAFKA_SERVERS,
        security_protocol="SASL_SSL",
        sasl_mechanism="SCRAM-SHA-512",
        sasl_plain_username='mlops2',
        sasl_plain_password='otus-mlops',
        ssl_cafile="CA.pem",
        fetch_min_bytes=100000,
        fetch_max_wait_ms=3000,
        session_timeout_ms=30000
    )

    producer = KafkaProducer(
        bootstrap_servers=KAFKA_SERVERS,
        security_protocol="SASL_SSL",
        sasl_mechanism="SCRAM-SHA-512",
        sasl_plain_username='mlops2',
        sasl_plain_password='otus-mlops',
        ssl_cafile="CA.pem")

    with open("model.pkl", "rb") as f:
        model = pickle.load(f)

    column_names = ["click_id", "ip", "app", "device", "os", "channel", "click_time"]

    while True:
        msg_pack = consumer.poll(1000, max_records=30000)
        if len(msg_pack) == 0:
            continue

        data = []
        
        for tp, lines in msg_pack.items():
            for line in lines:
                data.append(line.value.decode("utf-8").split(","))
        
        try:
            df = pd.DataFrame(data, columns=column_names)
            df.drop("click_id", axis=1, inplace=True)

            df = create_time_features(df)
            df = create_grouped_features(df)
        except ValueError:
            continue

        predictions = model.predict(df)
        predictions_str = "\n".join(map(str, predictions))
        
        producer.send('predictions', predictions_str.encode("utf-8"))
        producer.flush()
        

if __name__ == "__main__":
    main()

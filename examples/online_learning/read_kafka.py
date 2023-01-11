import mindpandas as pd
from kafka import KafkaConsumer
from kafka.structs import TopicPartition

def kafka_read(bootstrap_servers, topic_partitions, auto_offset_reset='latest', key_deserializer=None, value_deserializer=None, group_id='', api_version=(0, 10, 2), count=10):
    if isinstance(topic_partitions, str):
        consumer = KafkaConsumer(bootstrap_servers, topic_partitions, auto_offset_reset=auto_offset_reset,
                                 key_deserializer=key_deserializer, value_deserializer=value_deserializer, api_version=api_version, group_id=group_id)
    elif isinstance(topic_partitions, list):
        if isinstance(topic_partitions[0], str):
            consumer = KafkaConsumer(bootstrap_servers=bootstrap_servers, auto_offset_reset=auto_offset_reset,
                                     key_deserializer=key_deserializer, value_deserializer=value_deserializer, api_version=api_version, group_id=group_id)
            consumer.subscribe(topic_partitions)
        elif isinstance(topic_partitions[0], tuple):
            consumer = KafkaConsumer(bootstrap_servers=bootstrap_servers, auto_offset_reset=auto_offset_reset,
                                     key_deserializer=key_deserializer, value_deserializer=value_deserializer, api_version=api_version, group_id=group_id)
            tp = []
            for i in topic_partitions:
                if len(i) == 2:
                    tp.append(TopicPartition(i[0], i[1]))
                else:
                    raise ValueError("Input has to be (topic, partition) format.")
            consumer.assign(tp)
        else:
            raise ValueError("topic_partitions has to be either list of topics or list of tuple, which contains topic name and corresponsing partitions chosen.")
    else:
        raise ValueError("topic_partitions has to be either list of string or list of tuple")

    try:
        iter = 0
        news = []
        for msg in consumer:
            iter += 1
            news.append(msg.value)
            if iter >= count:
                df = pd.DataFrame(news, columns=msg.key)
                yield df
                news = []
                iter = 0
    except KeyboardInterrupt:
        raise Exception("Aborted by user...")

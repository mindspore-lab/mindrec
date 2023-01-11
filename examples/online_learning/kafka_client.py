from kafka import KafkaAdminClient
from kafka.admin import NewTopic

topic = "python_test1"
bootstrap_servers = 'localhost:9092'

admin_client = KafkaAdminClient(bootstrap_servers=bootstrap_servers)
topic_list = []
topic_list.append(NewTopic(name=topic, num_partitions=2, replication_factor=1))
admin_client.create_topics(new_topics=topic_list, validate_only=False)

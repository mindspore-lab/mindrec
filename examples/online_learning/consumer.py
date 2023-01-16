import pickle
import numpy as np
from read_kafka import kafka_read
import time
from mindpandas.channel import DataSender
import json
import argparse

def get_weight(x):
    ret = []
    for i, val in enumerate(x):
        if i < 13:
            if np.isnan(float(val)):
                ret.append(0)
            else:
                col = f'I{i + 1}'
                ret.append((float(val) - min_dict[col]) / (max_dict[col] - min_dict[col]))
        else:
            ret.append(1)
    return np.array(ret)

def get_id(x):
    ret = []
    for i, val in enumerate(x):
        if i < 13:
            ret.append(i + 1)
        else:
            key = f'OOVC{i - 12}'
            col = f'I{i + 1}'
            ret.append(map_dict.get(col, map_dict[key]))
    return np.array(ret)

def get_label(x):
    return np.array([int(float(x))])

def parse_args():
    parser = argparse.ArgumentParser(description='Consumer')
    parser.add_argument("--max_dict", type=str, default='./all_val_max_dict.pkl')
    parser.add_argument("--min_dict", type=str, default='./all_val_min_dict.pkl')
    parser.add_argument("--map_dict", type=str, default='./cat2id_dict.pkl')
    parser.add_argument("--address", type=str, default='127.0.0.1')
    parser.add_argument("--dataset_name", type=str, default='criteo')
    parser.add_argument("--namespace", type=str, default='demo')
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--window_size", type=int, default=1000)

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    max_dict = pickle.load(open(args.max_dict, "rb"))
    min_dict = pickle.load(open(args.min_dict, "rb"))
    map_dict = pickle.load(open(args.map_dict, "rb"))

    sender = DataSender(address=args.address,
                        dataset_name=args.dataset_name,
                        namespace=args.namespace,
                        num_shards=args.num_shards,
                        full_batch=False)
    myTopic = [('python_test1', 0), ('python_test1', 1)]
    myGroup = "test"
    deserializer = lambda v: json.loads(v.decode('utf-8'))
    print("Initialization done")
    count = args.num_shards * args.window_size
    for df in kafka_read(bootstrap_servers='localhost:9092',
                         topic_partitions=myTopic,
                         auto_offset_reset='latest',
                         key_deserializer=deserializer,
                         value_deserializer=deserializer,
                         group_id=myGroup,
                         api_version=(0, 10, 2),
                         count=count):
        features = df.drop(columns=['label'], axis=1)
        feat_id = features.apply(get_id, axis=1)
        feat_weight = features.apply(get_weight, axis=1)

        df['id'] = feat_id
        df['weight'] = feat_weight
        df['label'] = df['label'].apply(get_label)

        df = df[['id', 'weight', 'label']]
        sender.send(df)

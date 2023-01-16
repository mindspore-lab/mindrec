# Copyright 2022 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
import os
import pickle
import argparse
import logging
import collections
import pandas as pd
import numpy as np
import time
import math
import concurrent.futures as cf
from multiprocessing import Process, Queue

from mindspore.mindrecord import FileWriter


NUM_INTEGER_COLUMNS = 13
NUM_CATEGORICAL_COLUMNS = 26
NUM_TOTAL_COLUMNS = 1 + NUM_INTEGER_COLUMNS + NUM_CATEGORICAL_COLUMNS

CAT_COUNT_THRESHOLD = 6
PREPROCESS_WORKER_NUM = 24
PREPROCESS_CHUNKSIZE=5000000
SAVE_MINDRECORD_WORKER_NUM = 32
SAVE_MINDRECORD_CHUNKSIZE=1000000

logging.basicConfig(format='%(asctime)s %(message)s')
logging.root.setLevel(logging.INFO)

class StatsDict():
    """preprocessed data"""

    def __init__(self, dense_dim, slot_dim, skip_id_convert):
        self.field_size = dense_dim + slot_dim
        self.dense_dim = dense_dim
        self.slot_dim = slot_dim
        self.skip_id_convert = bool(skip_id_convert)

        self.val_cols = [idx2key(idx + 1) for idx in range(dense_dim)]
        self.cat_cols = [idx2key(idx + 1) for idx in range(dense_dim, dense_dim+slot_dim)]

        self.val_min_dict = {col: 0 for col in self.val_cols}
        self.val_max_dict = {col: 0 for col in self.val_cols}
        self.cat_unique_counts = {col: None for col in self.cat_cols}

        self.oov_prefix = "OOV"
        self.cat2id_dict = {}
        self.cat2id_dict.update({col: i for i, col in enumerate(self.val_cols)})
        self.cat2id_dict.update(
            {self.oov_prefix + col: i + len(self.val_cols) for i, col in enumerate(self.cat_cols)})


    def get_stats(self, col, colData):
        if col in self.val_cols:
            min_max = (np.int64(colData.min()), np.int64(colData.max()))
            if col != 'label':
                logging.info('column {} [{}, {}]'.format(col, str(min_max[0]), str(min_max[1])))
            self.val_min_dict[col] = min_max[0]
            self.val_max_dict[col] = min_max[1]

        if col in self.cat_cols:
            logging.info("start to get cat_unique_counts of col {}".format(col))
            self.cat_unique_counts[col] = colData.value_counts()
            logging.info("finish to get cat_unique_counts of col {}".format(col))

    def save_dict(self, dict_path, part, col, chunk_id):
        with open(os.path.join(dict_path, "{}_{}_{}_val_max_dict.pkl".format(part, chunk_id, col)), "wb") as file_wrt:
            pickle.dump(self.val_max_dict, file_wrt)
        with open(os.path.join(dict_path, "{}_{}_{}_val_min_dict.pkl".format(part, chunk_id, col)), "wb") as file_wrt:
            pickle.dump(self.val_min_dict, file_wrt)
        with open(os.path.join(dict_path, "{}_{}_{}_cat_unique_counts.pkl".format(part, chunk_id, col)), "wb") as file_wrt:
            pickle.dump(self.cat_unique_counts, file_wrt)

    def load_dict(self, dict_path, part_num, chunk_of_part):
        logging.info("load dict from {} of {} parts".format(dict_path, part_num))

        cat_count_dict = {col: collections.defaultdict(int) for col in self.cat_cols}
        for part in range(part_num):
            logging.info("load dict of part{}".format(part))
            chunk_num = chunk_of_part[part]
            for chunk_id in range(1, chunk_num+1):
                for col in self.val_cols:
                    logging.info("load min max of part-chunk-col: {}-{}-{}".format(part, chunk_id, col))
                    with open(os.path.join(dict_path, "{}_{}_{}_val_max_dict.pkl".format(part, chunk_id, col)), "rb") as file_wrt:
                        max_dict = pickle.load(file_wrt)
                        logging.debug("max_dict.items()[:50]:{}".format(list(max_dict.items())))
                        if max_dict[col] > self.val_max_dict[col]:
                            self.val_max_dict[col] = max_dict[col]
                    with open(os.path.join(dict_path, "{}_{}_{}_val_min_dict.pkl".format(part, chunk_id, col)), "rb") as file_wrt:
                        min_dict = pickle.load(file_wrt)
                        logging.debug("min_dict.items()[:50]:{}".format(list(min_dict.items())))
                        if min_dict[col] < self.val_min_dict[col]:
                            self.val_min_dict[col] = min_dict[col]

                for col in self.cat_cols:
                    logging.info("load cat of col: {}".format(col))
                    with open(os.path.join(dict_path, "{}_{}_{}_cat_unique_counts.pkl".format(part, chunk_id, col)), "rb") as file_wrt:
                        part_dict = pickle.load(file_wrt)
                        for cat, count in part_dict[col].items():
                            cat_count_dict[col][cat] += count
                        logging.info("after union, col cat_count_dict len: {}".format(len(cat_count_dict[col])))

        for col, cat_count_d in cat_count_dict.items():
            new_cat_count_d = dict(filter(lambda x: x[1] > CAT_COUNT_THRESHOLD, cat_count_d.items()))
            for cat_str, _ in new_cat_count_d.items():
                self.cat2id_dict[col + "_" + cat_str] = len(self.cat2id_dict)
        logging.info("val_min_dict.items():{}".format(list(self.val_min_dict.items())))
        logging.info("val_max_dict.items():{}".format(list(self.val_max_dict.items())))
        logging.info("cat2id_dict len:{}".format(len(self.cat2id_dict)))

        with open(os.path.join(dict_path, "all_val_max_dict.pkl"), "wb") as file_wrt:
            pickle.dump(self.val_max_dict, file_wrt)
        with open(os.path.join(dict_path, "all_val_min_dict.pkl"), "wb") as file_wrt:
            pickle.dump(self.val_min_dict, file_wrt)
        with open(os.path.join(dict_path, "cat2id_dict.pkl"), "wb") as file_wrt:
            pickle.dump(self.cat2id_dict, file_wrt)


    def map_cat2id(self, values, cats):
        """Cat to id"""
        def minmax_scale_value(i, val):
            max_v = float(self.val_max_dict[idx2key(i + 1)])
            return float(val) * 1.0 / max_v

        id_list = []
        weight_list = []
        for i, val in enumerate(values):
            if val == "" or math.isnan(val):
                id_list.append(i)
                weight_list.append(0)
            else:
                key = idx2key(i + 1)
                id_list.append(self.cat2id_dict[key])
                weight_list.append(minmax_scale_value(i, float(val)))

        for i, cat_str in enumerate(cats):
            key = idx2key(i+1+NUM_INTEGER_COLUMNS) + "_" + str(cat_str)
            if key in self.cat2id_dict:
                if self.skip_id_convert is True:
                    # For the synthetic data, if the generated id is between [0, max_vcoab], but the num examples is l
                    # ess than vocab_size/ slot_nums the id will still be converted to [0, real_vocab], where real_vocab
                    # the actually the vocab size, rather than the max_vocab. So a simple way to alleviate this
                    # problem is skip the id convert, regarding the synthetic data id as the final id.
                    id_list.append(cat_str)
                else:
                    id_list.append(self.cat2id_dict[key])
            else:
                id_list.append(self.cat2id_dict[self.oov_prefix + idx2key(i+1+NUM_INTEGER_COLUMNS)])
            weight_list.append(1.0)
        return id_list, weight_list

def _wait_futures_and_reset(futures):
    for future in futures:
        result = future.result()
    futures = list()

def preprocess(day_num, stats_save_path, dense_dim=13, slot_dim=26):
    #import pdb; pdb.set_trace()
    chunk_of_part = dict()
    with cf.ProcessPoolExecutor(max_workers=PREPROCESS_WORKER_NUM) as executor:
        futures = list()
        for part in range(day_num):
            logging.info('start preprocess part {}'.format(part))
            df = open_data_file_by_chunk(data_files[part], PREPROCESS_CHUNKSIZE)
            count = 0
            for chunk in df:
                count += 1
                logging.info('preprocess chunk {}'.format(count))
                for col, colData in chunk.iteritems():
                    logging.info("submit for part col {}-{} ".format(part, col))
                    future = executor.submit(stats_by_col, col, colData, part, count, stats_save_path, dense_dim, slot_dim)
                    futures.append(future)

            logging.info('finish part {}, total chunk {}'.format(part, count))
            chunk_of_part[part] = count

        _wait_futures_and_reset(futures)
    return chunk_of_part


def stats_by_col(col, colData, part, chunk_id, dict_path, dense_dim, slot_dim):
    #import pdb; pdb.set_trace()
    try:
        stats = StatsDict(dense_dim, slot_dim, 0)
        stats.get_stats(col, colData)
        stats.save_dict(dict_path, part, col, chunk_id)
    except Exception as e:
        logging.error(e)

def get_mindrecord_writer(pro_num, pro_type, output_dir):
    schema = {"label": {"type": "float32", "shape": [-1]},
              "feat_vals": {"type": "float32", "shape": [-1]},
              "feat_ids": {"type": "int32", "shape": [-1]}}

    name = os.path.join(output_dir, "{}_{}_input_part.mindrecord".format(pro_type, pro_num))
    mr_writer = FileWriter(name, 10)
    mr_writer.add_schema(schema, "CRITEO_TRAIN")

    return mr_writer, name


def write_to_mindrecord(input_q, pro_num, pro_type, output_dir, stats, dense_dim, slot_dim):
    feature_size = dense_dim + slot_dim

    line_per_sample = 1000
    part_rows = 100
    data_list = []
    ids_list = []
    wts_list = []
    label_list = []
    items_error_size_lineCount = []

    writer, mr_file_name = get_mindrecord_writer(pro_num, pro_type, output_dir)

    while 1:
        if input_q.empty():
            time.sleep(0.01)
        else:
            df = input_q.get()
            logging.info("{}-{} get a chunk".format(pro_num, pro_type))
            if isinstance(df, str):
                break

            count = 0
            for idx, row in df.iterrows():
                count += 1
                if count % 100000 == 0:
                    logging.info("Have handle {}w lines.".format(count // 10000))

                if len(row) != (1 + dense_dim + slot_dim):
                    items_error_size_lineCount.append(idx)
                    continue

                label = float(row[0])
                values = row[1:1 + dense_dim]
                cats = row[1 + dense_dim:]

                assert len(values) == dense_dim, "values.size: {}".format(len(values))
                assert len(cats) == slot_dim, "cats.size: {}".format(len(cats))

                ids, wts = stats.map_cat2id(values, cats)
                ids_list.extend(ids)
                wts_list.extend(wts)
                label_list.append(label)

                if count % line_per_sample == 0:
                    if len(ids_list) == line_per_sample * feature_size and len(wts_list) == line_per_sample * feature_size and len(label_list) == line_per_sample:
                        data_list.append({"feat_ids": np.array(ids_list, dtype=np.int32),
                                          "feat_vals": np.array(wts_list, dtype=np.float32),
                                          "label": np.array(label_list, dtype=np.float32)
                                          })
                    else:
                        logging.error("get a wrong data {}-{}-{}".format(len(ids_list), len(wts_list), len(label_list)))

                    ids_list.clear()
                    wts_list.clear()
                    label_list.clear()

                if data_list and len(data_list) % part_rows == 0:
                    writer.write_raw_data(data_list)
                    data_list.clear()
            logging.info("Finish to process a chunk")

            if data_list:
                # writer.write_raw_data(data_list)
                logging.info("drop some data:".format(len(data_list)))

    writer.commit()
    logging.info("==>> End to create mindrecord file: {}".format(mr_file_name))
    logging.info("items_error_size_lineCount.size(): {}.".format(len(items_error_size_lineCount)))


def save_mindrecord(day_num, test_size, stats, output_dir, dense_dim, slot_dim):
    logging.info('save to mindrecord')
    train_worker_num = SAVE_MINDRECORD_WORKER_NUM
    test_worker_num = max(1, int(test_size * train_worker_num))

    train_process_list = dict()
    train_input_q_list = dict()
    test_process_list = dict()
    test_input_q_list = dict()

    for i in range(train_worker_num):
        name = "train_p"+str(i)
        input_q = Queue(1)
        p = Process(name=name, target=write_to_mindrecord, args=(input_q, i, "train", output_dir, stats, dense_dim, slot_dim))
        p.start()
        train_process_list[name] = p
        train_input_q_list[name] = input_q
        logging.info("Start process: {}".format(name))
    for i in range(test_worker_num):
        name = "test_p"+str(i)
        input_q = Queue(1)
        p = Process(name=name, target=write_to_mindrecord, args=(input_q, i, "test", output_dir, stats, dense_dim, slot_dim))
        p.start()
        test_process_list[name] = p
        test_input_q_list[name] = input_q
        logging.info("Start process: {}".format(name))

    for part in range(day_num):
        df = open_data_file_by_chunk(data_files[part], SAVE_MINDRECORD_CHUNKSIZE)
        logging.info('start save mindrecord for part {}'.format(part))

        count = 0
        total_count = 0
        for chunk in df:
            count += 1
            total_count += 1

            if count * test_size >= 1:
                logging.info("submit chunk to test, chunk id: {}".format(total_count))
                put_chunk_to_queue(test_input_q_list, chunk)
                count = 0
            else:
                logging.info("submit chunk to train, chunk id: {}".format(total_count))
                put_chunk_to_queue(train_input_q_list, chunk)

    # waiting for all the data write success
    wait_process_exit(train_input_q_list, train_process_list)
    wait_process_exit(test_input_q_list, test_process_list)

def put_chunk_to_queue(q_list, chunk):
    put_queue_success = False
    while not put_queue_success:
        for filename in q_list:  # put the data to multi child process queue
            while not q_list[filename].full() and not put_queue_success:
                q_list[filename].put(chunk)
                logging.info("success to submit chunk to queue: {}".format(filename))
                put_queue_success = True
            time.sleep(0.01)

def wait_process_exit(q_list, p_list):
    # waiting for all the data write success
    for filename in q_list:
        while q_list[filename].full():
            time.sleep(0.01)
        q_list[filename].put("end")  # send end to the child process, child process will commit and exit
        logging.info("Submit pro: {}".format(filename))

        while p_list[filename].is_alive():  # wait child process exit
            time.sleep(0.01)
        logging.info("Exit pro: {}".format(filename))

def mkdir_path(file_path):
    if not os.path.exists(file_path):
        os.makedirs(file_path)
        logging.info("mkdir: " + file_path)


def idx2key(idx):
    if idx == 0:
        return 'label'
    return 'I' + str(idx) if idx <= NUM_INTEGER_COLUMNS else 'C' + str(idx - NUM_INTEGER_COLUMNS)


def open_data_file_by_chunk(data_file, chunksize):
    cols = [idx2key(idx) for idx in range(0, NUM_TOTAL_COLUMNS)]

    logging.info('start to read a CSV file: {}'.format(data_file))
    df = pd.read_csv(data_file, sep='\t', names=cols, chunksize=chunksize)
    logging.info('end to read a CSV file')
    return df


def parse_args():
    parser = argparse.ArgumentParser(description=("Criteo Dataset Preprocessing"))
    parser.add_argument("--data_path", type=str, help="Input dataset path (Required)")
    parser.add_argument("--part_num", type=int, help="Num of part of criteo data (Required)")

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()

    data_root_dir = args.data_path
    dense_dim = NUM_INTEGER_COLUMNS
    slot_dim = NUM_CATEGORICAL_COLUMNS
    target_field_size = dense_dim + slot_dim
    skip_id_convert = 0
    test_size = 0.1

    day_num = args.part_num
    data_files = [os.path.join(data_root_dir,  "day_" + str(i)) for i in range(day_num)]
    stats_save_path = os.path.join(data_root_dir, "stats_dict")
    mkdir_path(stats_save_path)
    mindrecord_save_path = os.path.join(data_root_dir, "mindrecord")
    mkdir_path(mindrecord_save_path)

    chunk_of_day = preprocess(day_num, stats_save_path, dense_dim, slot_dim)
    logging.info(chunk_of_day)

    all_stats = StatsDict(dense_dim, slot_dim, skip_id_convert)
    all_stats.load_dict(stats_save_path, day_num, chunk_of_day)

    save_mindrecord(day_num, test_size, all_stats, mindrecord_save_path, dense_dim, slot_dim)


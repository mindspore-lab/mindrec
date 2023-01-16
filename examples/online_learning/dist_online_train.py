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

import os
from mindspore import context
from mindspore_rec import RecModel as Model
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig, TimeMonitor
from src.wide_and_deep import PredictWithSigmoid, TrainStepWrap, NetWithLossClass, WideDeepModel
from src.callbacks import LossCallBack
from src.model_utils.config import config

import numpy as np
import mindpandas as mpd
import mindspore.dataset as ds
from mindpandas.channel import DataReceiver
from mindspore.train.serialization import save_checkpoint
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig, TimeMonitor

from mindspore.communication.management import get_rank, get_group_size, init
from mindspore.context import ParallelMode

class StreamingDataset:
    def __init__(self, receiver):
        self.data_ = []
        self.receiver_ = receiver

    def __getitem__(self, item):
        while not self.data_:
            data = self.receiver_.recv()
            if data is not None:
                self.data_ = data.tolist()

        last_row = self.data_.pop()
        return np.array(last_row[0], dtype=np.int32), np.array(last_row[1], dtype=np.float32), np.array(last_row[2], dtype=np.float32)

    def __len__(self):
        return 2**20 - 1

def GetWideDeepNet(configure):
    """
    Get network of wide&deep model.
    """
    WideDeep_net = WideDeepModel(configure)

    loss_net = NetWithLossClass(WideDeep_net, configure)
    train_net = TrainStepWrap(loss_net)
    eval_net = PredictWithSigmoid(WideDeep_net)

    return train_net, eval_net


if __name__ == '__main__':
    _enable_graph_kernel = config.device_target != "GPU"
    context.set_context(mode=context.GRAPH_MODE, save_graphs=True,
                        enable_graph_kernel=_enable_graph_kernel, device_target=config.device_target)
    if _enable_graph_kernel:
        context.set_context(graph_kernel_flags="--enable_cluster_ops=MatMul")

    init()
    context.set_context(save_graphs_path='./graphs_of_device_id_'+str(get_rank()))
    context.set_auto_parallel_context(parallel_mode=ParallelMode.DATA_PARALLEL, gradients_mean=True,
                                      device_num=get_group_size())

    
    import logging
    logging.basicConfig(level=logging.DEBUG)

    receiver = DataReceiver(address=config.address, namespace=config.namespace,
                            dataset_name=config.dataset_name, shard_id=get_rank())
    stream_dataset = StreamingDataset(receiver)

    dataset = ds.GeneratorDataset(stream_dataset, column_names=["id", "weight", "label"])
    dataset = dataset.batch(config.batch_size)

    train_net, _ = GetWideDeepNet(config)
    train_net.set_train()

    model = Model(train_net)
    callback = LossCallBack(config=config) 

    # Ckpt
    ckptconfig = CheckpointConfig(save_checkpoint_steps=100, keep_checkpoint_max=5)
    ckpoint_cb = ModelCheckpoint(prefix='widedeep_train', directory="./ckpt"+str(get_rank()), config=ckptconfig)

    # Start train
    model.online_train(dataset, callbacks=[TimeMonitor(1), callback, ckpoint_cb], dataset_sink_mode=True)

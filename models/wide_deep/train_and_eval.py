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

from mindspore import Model, context
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig, TimeMonitor

from src.wide_and_deep import PredictWithSigmoid, TrainStepWrap, NetWithLossClass, WideDeepModel
from src.callbacks import LossCallBack, EvalCallBack
from src.datasets import create_dataset, DataType
from src.metrics import AUCMetric
from src.model_utils.config import config as cfg


def get_wide_deep_net(config):
    """
    Get network of wide&deep model.
    """
    wide_deep_net = WideDeepModel(config)

    loss_net = NetWithLossClass(wide_deep_net, config)
    train_net = TrainStepWrap(loss_net, sparse=config.sparse, dynamic_embedding=config.dynamic_embedding)
    eval_net = PredictWithSigmoid(wide_deep_net)

    return train_net, eval_net


class ModelBuilder():
    """
    ModelBuilder
    """

    def __init__(self):
        pass

    def get_hook(self):
        pass

    def get_train_hook(self):
        hooks = []
        callback = LossCallBack()
        hooks.append(callback)

        if int(os.getenv('DEVICE_ID')) == 0:
            pass
        return hooks

    def get_net(self, config):
        return get_wide_deep_net(config)


def test_train_eval(config):
    """
    test_train_eval
    """
    data_path = config.data_path
    batch_size = config.batch_size
    epochs = config.epochs
    sparse = config.sparse
    if config.dataset_type == "tfrecord":
        dataset_type = DataType.TFRECORD
    elif config.dataset_type == "mindrecord":
        dataset_type = DataType.MINDRECORD
    else:
        dataset_type = DataType.H5
    ds_train = create_dataset(data_path, train_mode=True,
                              batch_size=batch_size, data_type=dataset_type)
    ds_eval = create_dataset(data_path, train_mode=False,
                             batch_size=batch_size, data_type=dataset_type)
    print("ds_train.size: {}".format(ds_train.get_dataset_size()))
    print("ds_eval.size: {}".format(ds_eval.get_dataset_size()))

    net_builder = ModelBuilder()

    train_net, eval_net = net_builder.get_net(config)
    train_net.set_train()
    auc_metric = AUCMetric()

    model = Model(train_net, eval_network=eval_net, metrics={"auc": auc_metric})

    eval_callback = EvalCallBack(model, ds_eval, auc_metric, config)

    callback = LossCallBack(config=config)
    ckptconfig = CheckpointConfig(save_checkpoint_steps=ds_train.get_dataset_size(), keep_checkpoint_max=5)
    ckpoint_cb = ModelCheckpoint(prefix='widedeep_train', directory=config.ckpt_path, config=ckptconfig)

    model.train(epochs, ds_train,
                callbacks=[TimeMonitor(ds_train.get_dataset_size()), eval_callback, callback, ckpoint_cb],
                dataset_sink_mode=True)


def train_wide_and_deep():
    if cfg.device_target == "Ascend":
        context.set_context(ascend_config={"op_precision_mode": "op_precision.ini"})
    _enable_graph_kernel = cfg.device_target == "GPU"
    context.set_context(mode=context.GRAPH_MODE,
                        enable_graph_kernel=_enable_graph_kernel, device_target=cfg.device_target,
                        save_graphs=True, save_graphs_path="./graphs")
    if _enable_graph_kernel:
        context.set_context(graph_kernel_flags="--enable_cluster_ops=MatMul")
    test_train_eval(cfg)


if __name__ == "__main__":
    train_wide_and_deep()

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
from src.callbacks import LossCallBack
from src.datasets import create_dataset, DataType
from src.model_utils.config import config


def get_wide_deep_net(configure):
    """
    Get network of wide&deep model.
    """
    wide_deep_net = WideDeepModel(configure)

    loss_net = NetWithLossClass(wide_deep_net, configure)
    train_net = TrainStepWrap(loss_net, dynamic_embedding=config.dynamic_embedding)
    eval_net = PredictWithSigmoid(wide_deep_net)

    return train_net, eval_net


class ModelBuilder():
    """
    Build the model.
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

    def get_net(self, configure):
        return get_wide_deep_net(configure)


def test_train(configure):
    """
    test_train
    """
    data_path = configure.data_path
    batch_size = configure.batch_size
    epochs = configure.epochs
    if configure.dataset_type == "tfrecord":
        dataset_type = DataType.TFRECORD
    elif configure.dataset_type == "mindrecord":
        dataset_type = DataType.MINDRECORD
    else:
        dataset_type = DataType.H5
    ds_train = create_dataset(data_path, train_mode=True,
                              batch_size=batch_size, data_type=dataset_type)
    print("ds_train.size: {}".format(ds_train.get_dataset_size()))

    net_builder = ModelBuilder()
    train_net, _ = net_builder.get_net(configure)
    train_net.set_train()

    model = Model(train_net)
    callback = LossCallBack(config=configure)
    ckptconfig = CheckpointConfig(save_checkpoint_steps=ds_train.get_dataset_size(),
                                  keep_checkpoint_max=5)
    ckpoint_cb = ModelCheckpoint(prefix='widedeep_train', directory=configure.ckpt_path, config=ckptconfig)
    model.train(epochs, ds_train, callbacks=[TimeMonitor(ds_train.get_dataset_size()), callback, ckpoint_cb])


def train_wide_and_deep():
    if cfg.device_target == "Ascend":
        context.set_context(ascend_config={"op_precision_mode": "op_precision.ini"})
    _enable_graph_kernel = config.device_target == "GPU"
    context.set_context(mode=context.GRAPH_MODE,
                        enable_graph_kernel=_enable_graph_kernel, device_target=config.device_target)
    if _enable_graph_kernel:
        context.set_context(graph_kernel_flags="--enable_cluster_ops=MatMul")
    test_train(config)


if __name__ == "__main__":
    train_wide_and_deep()

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
"""train standalone on parameter server."""


import os
import sys
from mindspore import Model, context
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig, TimeMonitor
from mindspore.common import set_seed

from mindspore.communication.management import init
from src.wide_and_deep import PredictWithSigmoid, TrainStepWrap, NetWithLossClass, WideDeepModel
from src.callbacks import LossCallBack, EvalCallBack
from src.datasets import create_dataset, DataType
from src.metrics import AUCMetric
from src.model_utils.config import config as cfg

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def get_wide_deep_net(config):
    """
    Get network of wide&deep model.
    """
    wide_deep_net = WideDeepModel(config)
    loss_net = NetWithLossClass(wide_deep_net, config)
    train_net = TrainStepWrap(loss_net, parameter_server=bool(config.parameter_server),
                              sparse=config.sparse, cache_enable=(config.vocab_cache_size > 0),
                              dynamic_embedding=config.dynamic_embedding)
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


def train_and_eval(config):
    """
    test_train_eval
    """
    set_seed(1000)
    data_path = config.data_path
    batch_size = config.batch_size
    epochs = config.epochs
    if config.dataset_type == "tfrecord":
        dataset_type = DataType.TFRECORD
    elif config.dataset_type == "mindrecord":
        dataset_type = DataType.MINDRECORD
    else:
        dataset_type = DataType.H5
    parameter_server = bool(config.parameter_server)
    print("epochs is {}".format(epochs))
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
    ms_role = os.getenv("MS_ROLE")
    if ms_role == "MS_WORKER":
        if cache_enable:
            ckptconfig = CheckpointConfig(save_checkpoint_steps=ds_train.get_dataset_size() * epochs,
                                          keep_checkpoint_max=1)
        else:
            ckptconfig = CheckpointConfig(save_checkpoint_steps=ds_train.get_dataset_size(), keep_checkpoint_max=5)
    else:
        ckptconfig = CheckpointConfig(save_checkpoint_steps=1, keep_checkpoint_max=1)
    ckpoint_cb = ModelCheckpoint(prefix='widedeep_train', directory=config.ckpt_path, config=ckptconfig)
    callback_list = [TimeMonitor(ds_train.get_dataset_size()), eval_callback, callback, ckpoint_cb]

    model.train(epochs, ds_train,
                callbacks=callback_list,
                dataset_sink_mode=(parameter_server and cache_enable))


context.set_context(mode=context.GRAPH_MODE, device_target=cfg.device_target, save_graphs=True)
cache_enable = cfg.vocab_cache_size > 0

def train_wide_and_deep():
    """ train_wide_and_deep """
    if cfg.device_target == "Ascend":
        context.set_context(ascend_config={"op_precision_mode": "op_precision.ini"})
    context.set_ps_context(enable_ps=True)
    init()

    if not cache_enable:
        cfg.sparse = True
    if cfg.device_target == "GPU":
        context.set_context(enable_graph_kernel=True)
        context.set_context(graph_kernel_flags="--enable_cluster_ops=MatMul")

    train_and_eval(cfg)


if __name__ == "__main__":
    train_wide_and_deep()

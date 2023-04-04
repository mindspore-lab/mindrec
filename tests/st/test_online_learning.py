# Copyright 2023 Huawei Technologies Co., Ltd
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
"""Online learning st"""
import mindspore.dataset as ds
import numpy as np
import pytest
from mindspore import context, nn
from mindspore.common.initializer import initializer
from mindspore.common.parameter import Parameter
from mindspore.ops import operations as P

from mindspore_rec import RecModel as Model


class Net(nn.Cell):
    """Backbone network"""

    def __init__(self):
        super().__init__()
        self.embeddinglookup = P.EmbeddingLookup()
        self.vocab_size = 200
        self.embedding_size = 80
        self.embedding_table = Parameter(
            initializer("normal", [self.vocab_size, self.embedding_size]),
            name="embedding_table",
        )
        self.gatherv2 = P.Gather()

    def construct(self, indices):
        out = self.gatherv2(self.embedding_table, indices, 0)
        return out


class StreamingDataset:
    def __init__(self):
        self.data = []

    def __getitem__(self, item):
        return np.ones((39,)).astype(np.int32)

    def __len__(self):
        return 2**20 - 1


def test_online_learning_api_sink_size_is_negative():
    """
    Feature: test online learning api.
    Description: enable data sink mode, and sink_size=-1.
    Expectation: raise a ValueError.
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")

    stream_dataset = StreamingDataset()
    dataset = ds.GeneratorDataset(
        stream_dataset, column_names=["id", "weight", "label"]
    )
    dataset = dataset.batch(100)

    train_net = Net()
    train_net.set_train()
    model = Model(train_net)

    with pytest.raises(ValueError) as exc_info:
        model.online_train(dataset, dataset_sink_mode=True, sink_size=-1)
    assert "The input value must be int and must > 0" in str(exc_info.value)


def test_online_learning_api_sink_size_not_equal_one():
    """
    Feature: test online learning api.
    Description: enable data sink mode, and sink_size=100.
    Expectation: raise a ValueError.
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")

    stream_dataset = StreamingDataset()
    dataset = ds.GeneratorDataset(
        stream_dataset, column_names=["id", "weight", "label"]
    )
    dataset = dataset.batch(100)

    train_net = Net()
    train_net.set_train()
    model = Model(train_net)

    with pytest.raises(ValueError) as exc_info:
        model.online_train(dataset, dataset_sink_mode=True, sink_size=100)
    assert "The sink_size parameter only support value of 1" in str(exc_info.value)


def test_online_learning_api_data_sink_mode_not_bool():
    """
    Feature: test online learning api.
    Description: enable data sink mode, and sink_size=100.
    Expectation: raise a ValueError.
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")

    stream_dataset = StreamingDataset()
    dataset = ds.GeneratorDataset(
        stream_dataset, column_names=["id", "weight", "label"]
    )
    dataset = dataset.batch(100)

    train_net = Net()
    train_net.set_train()
    model = Model(train_net)

    with pytest.raises(TypeError) as exc_info:
        model.online_train(dataset, dataset_sink_mode="valid")
    assert "The input value must be a bool, but got str" in str(exc_info.value)

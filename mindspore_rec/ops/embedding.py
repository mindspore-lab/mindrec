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

import sys
import mindspore as ms
import mindspore.common.dtype as mstype
from mindspore import log as logger
from mindspore.common.tensor import Tensor
from mindspore.ops import operations as P
from mindspore.ops import functional as F
from mindspore.common.parameter import Parameter
from mindspore.common.parameter import _get_unique_parameter_key
from mindspore.common.initializer import initializer
from mindspore.communication.management import get_group_size, get_rank
from mindspore.context import ParallelMode
from mindspore.parallel._utils import _get_parallel_mode, _get_full_batch
from mindspore.parallel._ps_context import _insert_hash_table_size, _set_cache_enable, \
                                           _set_cache_size, _set_sparse_format
from mindspore._checkparam import Rel
from mindspore._checkparam import Validator as validator
from mindspore.ops.primitive import constexpr
from mindspore.nn.layer.basic import ClipByNorm
from mindspore.nn.layer.math import Range
from mindspore.nn.cell import Cell
from mindspore.experimental import MapParameter


@constexpr
def _make_axis_range(start, end):
    axis = tuple(range(start, end))
    return axis


class HashEmbeddingLookup(Cell):
    r"""
    HashEmbeddingLookup layer.
    HashEmbeddingLookup is a dynamic version of EmbeddingLookup. The EmbeddingTable of the EmbeddingLookup
    operator is a Tensor type, the shape is fixed, does not support the dynamic addition and deletion of
    Embedding, and indices need to be encoded from 0 (indices stands for the line number of EmbeddingTable).
    The EmbeddingTable of HashEmbeddingLookup is a MapTensor type, and its internal is a HashMap structure,
    which supports the dynamic addition and deletion of Embedding, indices do not need to be encoded from 0,
    just any integers (except for -1, -2 two values), indices means keys of embedding features rather than the
    line number of EmbeddingTable, in addition, HashEmbeddingLookup also supports the permission and eviction 
    of features. This capability is common in recommender networks.

    Args:
        embedding_size (int): The size of each embedding vector.
        param_init (Union[Tensor, str, Initializer, numbers.Number]): Initializer for the embedding_table.
            Refer to class `initializer` for the values of string when a string
            is specified. Default: 'normal'.
        sparse (bool): Using sparse mode. Default: True.
        max_norm (Union[float, None]): A maximum clipping value. The data type must be float16, float32
                                       or None. Default: None
        permit_filter_value (numbers.Number): The permit filter value number. Default: 1.
        evict_filter_value (numbers.Number): The evict filter value number. Default: MAX_SIZE.
        vocab_cache_size (int): Cache size of the dictionary of embeddings. Default: 0. It is valid only in
            parameter server trainning mode. And the moment parameter of corresponding
            optimizer will also be set to the cache size. In addition, it should be noted that it will cost the device
            memory, so suggests setting a reasonable value to avoid insufficient memory.

    Inputs:
        - **input_indices** (Tensor) - The shape of tensor is :math:`(y_1, y_2, ..., y_S)`.
        indices do not need to be encoded from 0, could be int32 or int64 integer type (except for -1, -2 two values).

    Outputs:
        Tensor, the shape of tensor is :math:`(y_1, y_2, ..., y_S, embeddig_size)`.

    Supported Platforms:
        ``GPU``
    """

class HashEmbeddingLookup(Cell):
    def __init__(self, embedding_size, key_dtype=ms.int32, param_init='normal', sparse=True, max_norm=None,
                 permit_filter_value=1, evict_filter_value=sys.maxsize, vocab_cache_size=0):
        """Initialize HashEmbeddingLookup."""
        super(HashEmbeddingLookup, self).__init__()
        validator.check_value_type('sparse', sparse, [bool], self.cls_name)
        vocab_cache_size = validator.check_non_negative_int(vocab_cache_size, 'vocab_cache_size')

        self.forward_unique = sparse
        self.embedding_size = validator.check_positive_int(embedding_size, 'embedding_size', self.cls_name)
        self.embedding_table = MapParameter(key_dtype=key_dtype, value_dtype=ms.float32, value_shape=(embedding_size,),
                                            default_value=param_init, name='embedding_table',
                                            permit_filter_value=permit_filter_value,
                                            evict_filter_value=evict_filter_value)

        # Embedding cache mode for dynamic hash table.
        enable_cache = vocab_cache_size > 0
        if enable_cache:
            _set_cache_enable(True)
            _set_cache_size(vocab_cache_size)
            _set_sparse_format(True)
            self.embedding_table.cache_enable = True
            param_key = _get_unique_parameter_key()
            self.embedding_table.key = param_key
            _insert_hash_table_size(self.embedding_table.name, vocab_cache_size, embedding_size, 1, param_key)

        # Ops for sparse mode.
        self.gather_revert = P.Gather()
        self.reshape_first = P.Reshape()
        self.reshape = P.Reshape()
        self.unique = P.Unique()
        self.shape = P.Shape()
        aAAs =  ''
        self.embedding_table.unique = self.forward_unique

        self.max_norm = max_norm
        if self.max_norm is not None:
            self.max_norm = validator.check_positive_float(self.max_norm, 'max_norm', self.cls_name)
            self.max_norm = Tensor(self.max_norm, dtype=mstype.float32)

    def construct(self, indices):
        if self.forward_unique:
            shape = self.shape(indices) + (self.embedding_size,)
            indices_flatten = self.reshape_first(indices, (-1,))
            unique_id, unique_idx = self.unique(indices_flatten)
            weight_unique = self.embedding_table.get(unique_id)
            weight_flatten = self.gather_revert(weight_unique, unique_idx, 0)
            out = self.reshape(weight_flatten, shape)
        else:
            out = self.embedding_table.get(indices)

        if self.max_norm is not None:
            axis = _make_axis_range(F.rank(indices), F.rank(out))
            clip_by_norm = ClipByNorm(axis)
            out = clip_by_norm(out, self.max_norm)
        return out

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
"""wide and deep model"""
import mindspore.common.dtype as mstype
import numpy as np
from mindspore import Parameter, ParameterTuple, context, nn, ops
from mindspore.common.initializer import Uniform, initializer
from mindspore.communication.management import get_group_size
from mindspore.context import ParallelMode
from mindspore.nn import Dropout
from mindspore.nn.optim import FTRL, Adam, LazyAdam
from mindspore.nn.wrap.grad_reducer import DistributedGradReducer

np_type = np.float32
ms_type = mstype.float32


def init_method(method, shape, name, max_val=1.0):
    """
    parameter init method
    """
    if method in ["uniform"]:
        params = Parameter(initializer(Uniform(max_val), shape, ms_type), name=name)
    elif method == "one":
        params = Parameter(initializer("ones", shape, ms_type), name=name)
    elif method == "zero":
        params = Parameter(initializer("zeros", shape, ms_type), name=name)
    elif method == "normal":
        params = Parameter(initializer("normal", shape, ms_type), name=name)
    return params


def init_var_dict(init_args, in_vars):
    """
    var init function
    """
    var_map = {}
    _, max_val = init_args
    for _, item in enumerate(in_vars):
        key, shape, method = item
        # pylint: disable=C0201
        if key not in var_map.keys():
            if method in ["random", "uniform"]:
                var_map[key] = Parameter(
                    initializer(Uniform(max_val), shape, ms_type), name=key
                )
            elif method == "one":
                var_map[key] = Parameter(initializer("ones", shape, ms_type), name=key)
            elif method == "zero":
                var_map[key] = Parameter(initializer("zeros", shape, ms_type), name=key)
            elif method == "normal":
                var_map[key] = Parameter(
                    initializer("normal", shape, ms_type), name=key
                )
    return var_map


class DenseLayer(nn.Cell):
    """
    Dense Layer for Deep Layer of WideDeep Model;
    Containing: activation, matmul, bias_add;
    Args:
    """

    def __init__(
        self,
        input_dim,
        output_dim,
        weight_bias_init,
        act_str,
        keep_prob=0.5,
        use_activation=True,
        convert_dtype=True,
        drop_out=False,
    ):
        super().__init__()
        weight_init, bias_init = weight_bias_init
        self.weight = init_method(weight_init, [input_dim, output_dim], name="weight")
        self.bias = init_method(bias_init, [output_dim], name="bias")
        self.act_func = self._init_activation(act_str)
        self.matmul = ops.MatMul(transpose_b=False)
        self.bias_add = ops.BiasAdd()
        self.cast = ops.Cast()
        self.dropout = Dropout(p=1 - keep_prob)
        self.use_activation = use_activation
        self.convert_dtype = convert_dtype
        self.drop_out = drop_out

    def _init_activation(self, act_str):
        act_str = act_str.lower()
        if act_str == "relu":
            act_func = ops.ReLU()
        elif act_str == "sigmoid":
            act_func = ops.Sigmoid()
        elif act_str == "tanh":
            act_func = ops.Tanh()
        return act_func

    def construct(self, x):
        """
        Construct Dense layer
        """
        if self.training and self.drop_out:
            x = self.dropout(x)
        if self.convert_dtype:
            x = self.cast(x, mstype.float16)
            weight = self.cast(self.weight, mstype.float16)
            bias = self.cast(self.bias, mstype.float16)
            wx = self.matmul(x, weight)
            wx = self.bias_add(wx, bias)
            if self.use_activation:
                wx = self.act_func(wx)
            wx = self.cast(wx, mstype.float32)
        else:
            wx = self.matmul(x, self.weight)
            wx = self.bias_add(wx, self.bias)
            if self.use_activation:
                wx = self.act_func(wx)
        return wx


class WideDeepModel(nn.Cell):
    """
    From paper: " Wide & Deep Learning for Recommender Systems"
    Args:
        config (Class): The default config of Wide&Deep
    """

    def __init__(self, config):
        super().__init__()
        self.batch_size = config.batch_size
        parameter_server = bool(config.parameter_server)
        parallel_mode = context.get_auto_parallel_context("parallel_mode")
        is_auto_parallel = parallel_mode in (
            ParallelMode.SEMI_AUTO_PARALLEL,
            ParallelMode.AUTO_PARALLEL,
        )
        if is_auto_parallel:
            self.batch_size = self.batch_size * get_group_size()
        sparse = config.sparse
        self.field_size = config.field_size
        self.emb_dim = config.emb_dim
        self.weight_init, self.bias_init = config.weight_bias_init
        self.deep_input_dims = self.field_size * self.emb_dim
        self.all_dim_list = [self.deep_input_dims] + config.deep_layer_dim + [1]
        init_acts = [("Wide_b", [1], config.emb_init)]
        var_map = init_var_dict(config.init_args, init_acts)
        self.wide_b = var_map["Wide_b"]
        self.dense_layer_1 = DenseLayer(
            self.all_dim_list[0],
            self.all_dim_list[1],
            config.weight_bias_init,
            config.deep_layer_act,
            convert_dtype=config.use_mixed_precision,
            drop_out=config.dropout_flag,
        )
        self.dense_layer_2 = DenseLayer(
            self.all_dim_list[1],
            self.all_dim_list[2],
            config.weight_bias_init,
            config.deep_layer_act,
            convert_dtype=config.use_mixed_precision,
            drop_out=config.dropout_flag,
        )
        self.dense_layer_3 = DenseLayer(
            self.all_dim_list[2],
            self.all_dim_list[3],
            config.weight_bias_init,
            config.deep_layer_act,
            convert_dtype=config.use_mixed_precision,
            drop_out=config.dropout_flag,
        )
        self.dense_layer_4 = DenseLayer(
            self.all_dim_list[3],
            self.all_dim_list[4],
            config.weight_bias_init,
            config.deep_layer_act,
            convert_dtype=config.use_mixed_precision,
            drop_out=config.dropout_flag,
        )
        self.dense_layer_5 = DenseLayer(
            self.all_dim_list[4],
            self.all_dim_list[5],
            config.weight_bias_init,
            config.deep_layer_act,
            use_activation=False,
            convert_dtype=config.use_mixed_precision,
            drop_out=config.dropout_flag,
        )
        self.wide_mul = ops.Mul()
        self.deep_mul = ops.Mul()
        self.reduce_sum = ops.ReduceSum(keep_dims=False)
        self.reshape = ops.Reshape()
        self.deep_reshape = ops.Reshape()
        self.square = ops.Square()
        self.concat = ops.Concat(axis=1)
        self.unique = ops.Unique().shard(((1,),))
        self.wide_gatherv2 = ops.Gather()
        self.deep_gatherv2 = ops.Gather()
        if parameter_server:
            cache_enable = config.vocab_cache_size > 0
            target = "DEVICE" if cache_enable else "CPU"
            if not cache_enable:
                sparse = True
            if is_auto_parallel and config.full_batch and cache_enable:
                self.deep_embeddinglookup = nn.EmbeddingLookup(
                    config.vocab_size,
                    self.emb_dim,
                    target=target,
                    slice_mode=nn.EmbeddingLookup.TABLE_ROW_SLICE,
                    sparse=sparse,
                    vocab_cache_size=config.vocab_cache_size,
                )
                self.wide_embeddinglookup = nn.EmbeddingLookup(
                    config.vocab_size,
                    1,
                    target=target,
                    slice_mode=nn.EmbeddingLookup.TABLE_ROW_SLICE,
                    sparse=sparse,
                    vocab_cache_size=config.vocab_cache_size,
                )
            else:
                self.deep_embeddinglookup = nn.EmbeddingLookup(
                    config.vocab_size,
                    self.emb_dim,
                    target=target,
                    sparse=sparse,
                    vocab_cache_size=config.vocab_cache_size,
                )
                self.wide_embeddinglookup = nn.EmbeddingLookup(
                    config.vocab_size,
                    1,
                    target=target,
                    sparse=sparse,
                    vocab_cache_size=config.vocab_cache_size,
                )
            self.embedding_table = self.deep_embeddinglookup.embedding_table
            self.deep_embeddinglookup.embedding_table.set_param_ps()
            self.wide_embeddinglookup.embedding_table.set_param_ps()
        else:
            self.deep_embeddinglookup = nn.EmbeddingLookup(
                config.vocab_size,
                self.emb_dim,
                target="DEVICE",
                sparse=sparse,
                vocab_cache_size=config.vocab_cache_size,
            )
            self.wide_embeddinglookup = nn.EmbeddingLookup(
                config.vocab_size,
                1,
                target="DEVICE",
                sparse=sparse,
                vocab_cache_size=config.vocab_cache_size,
            )
            self.embedding_table = self.deep_embeddinglookup.embedding_table

    def construct(self, id_hldr, wt_hldr):
        """
        Args:
            id_hldr: batch ids;
            wt_hldr: batch weights;
        """
        # Wide layer
        wide_id_weight = self.wide_embeddinglookup(id_hldr)
        # Deep layer
        deep_id_embs = self.deep_embeddinglookup(id_hldr)
        mask = self.reshape(wt_hldr, (self.batch_size, self.field_size, 1))
        # Wide layer
        wx = self.wide_mul(wide_id_weight, mask)
        wide_out = self.reshape(self.reduce_sum(wx, 1) + self.wide_b, (-1, 1))
        # Deep layer
        vx = self.deep_mul(deep_id_embs, mask)
        deep_in = self.deep_reshape(vx, (-1, self.field_size * self.emb_dim))
        deep_in = self.dense_layer_1(deep_in)
        deep_in = self.dense_layer_2(deep_in)
        deep_in = self.dense_layer_3(deep_in)
        deep_in = self.dense_layer_4(deep_in)
        deep_out = self.dense_layer_5(deep_in)
        out = wide_out + deep_out
        return out, self.embedding_table


class NetWithLossClass(nn.Cell):

    """ "
    Provide WideDeep training loss through network.
    Args:
        network (Cell): The training network
        config (Class): WideDeep config
    """

    def __init__(self, network, config):
        super().__init__(auto_prefix=False)
        parameter_server = bool(config.parameter_server)
        sparse = config.sparse
        parallel_mode = context.get_auto_parallel_context("parallel_mode")
        is_auto_parallel = parallel_mode in (
            ParallelMode.SEMI_AUTO_PARALLEL,
            ParallelMode.AUTO_PARALLEL,
        )
        self.no_l2loss = parameter_server
        if sparse:
            self.no_l2loss = True
        self.network = network
        self.l2_coef = config.l2_coef
        self.loss = ops.SigmoidCrossEntropyWithLogits()
        self.square = ops.Square()
        self.reducemean_false = ops.ReduceMean(keep_dims=False)
        if is_auto_parallel:
            self.reducemean_false.add_prim_attr("cross_batch", True)
        self.reducesum_false = ops.ReduceSum(keep_dims=False)

    def construct(self, batch_ids, batch_wts, label):
        """
        Construct NetWithLossClass
        """
        predict, embedding_table = self.network(batch_ids, batch_wts)
        log_loss = self.loss(predict, label)
        wide_loss = self.reducemean_false(log_loss)
        if self.no_l2loss:
            deep_loss = wide_loss
        else:
            l2_loss_v = self.reducesum_false(self.square(embedding_table)) / 2
            deep_loss = self.reducemean_false(log_loss) + self.l2_coef * l2_loss_v

        return wide_loss, deep_loss


class IthOutputCell(nn.Cell):
    def __init__(self, network, output_index):
        super().__init__()
        self.network = network
        self.output_index = output_index

    def construct(self, x1, x2, x3):
        predict = self.network(x1, x2, x3)[self.output_index]
        return predict


class TrainStepWrap(nn.Cell):
    """
    Encapsulation class of WideDeep network training.
    Append Adam and FTRL optimizers to the training network after that construct
    function can be called to create the backward graph.
    Args:
        network (Cell): The training network. Note that loss function should have been added.
        sens (Number): The adjust parameter. Default: 1024.0
        parameter_server (Bool): Whether run in parameter server mode. Default: False
    """

    def __init__(
        self,
        network,
        sens=1024.0,
        parameter_server=False,
        sparse=False,
        cache_enable=False,
    ):
        super().__init__()
        parallel_mode = context.get_auto_parallel_context("parallel_mode")
        is_auto_parallel = parallel_mode in (
            ParallelMode.SEMI_AUTO_PARALLEL,
            ParallelMode.AUTO_PARALLEL,
        )
        self.network = network
        self.network.set_train()
        self.trainable_params = network.trainable_params()
        weights_w = []
        weights_d = []
        for params in self.trainable_params:
            if "wide" in params.name:
                weights_w.append(params)
            else:
                weights_d.append(params)
        self.weights_w = ParameterTuple(weights_w)
        self.weights_d = ParameterTuple(weights_d)

        if (sparse and is_auto_parallel) or (sparse and parameter_server):
            self.optimizer_d = LazyAdam(
                self.weights_d, learning_rate=3.5e-4, eps=1e-8, loss_scale=sens
            )
            self.optimizer_w = FTRL(
                learning_rate=5e-2,
                params=self.weights_w,
                l1=1e-8,
                l2=1e-8,
                initial_accum=1.0,
                loss_scale=sens,
            )
            if parameter_server and not cache_enable:
                self.optimizer_w.target = "CPU"
                self.optimizer_d.target = "CPU"
        else:
            self.optimizer_d = Adam(
                self.weights_d, learning_rate=3.5e-4, eps=1e-8, loss_scale=sens
            )
            self.optimizer_w = FTRL(
                learning_rate=5e-2,
                params=self.weights_w,
                l1=1e-8,
                l2=1e-8,
                initial_accum=1.0,
                loss_scale=sens,
            )
        self.hyper_map = ops.HyperMap()
        self.grad_w = ops.GradOperation(get_by_list=True, sens_param=True)
        self.grad_d = ops.GradOperation(get_by_list=True, sens_param=True)
        self.sens = sens
        self.loss_net_w = IthOutputCell(network, output_index=0)
        self.loss_net_d = IthOutputCell(network, output_index=1)
        self.loss_net_w.set_grad()
        self.loss_net_d.set_grad()

        self.reducer_flag = False
        self.grad_reducer_w = None
        self.grad_reducer_d = None
        self.reducer_flag = parallel_mode in (
            ParallelMode.DATA_PARALLEL,
            ParallelMode.HYBRID_PARALLEL,
        )
        if self.reducer_flag:
            mean = context.get_auto_parallel_context("gradients_mean")
            degree = context.get_auto_parallel_context("device_num")
            self.grad_reducer_w = DistributedGradReducer(
                self.optimizer_w.parameters, mean, degree
            )
            self.grad_reducer_d = DistributedGradReducer(
                self.optimizer_d.parameters, mean, degree
            )

    def construct(self, batch_ids, batch_wts, label):
        """
        Construct wide and deep model
        """
        weights_w = self.weights_w
        weights_d = self.weights_d
        loss_w, loss_d = self.network(batch_ids, batch_wts, label)
        sens_w = ops.Fill()(ops.DType()(loss_w), ops.Shape()(loss_w), self.sens)
        sens_d = ops.Fill()(ops.DType()(loss_d), ops.Shape()(loss_d), self.sens)
        grads_w = self.grad_w(self.loss_net_w, weights_w)(
            batch_ids, batch_wts, label, sens_w
        )
        grads_d = self.grad_d(self.loss_net_d, weights_d)(
            batch_ids, batch_wts, label, sens_d
        )
        if self.reducer_flag:
            grads_w = self.grad_reducer_w(grads_w)
            grads_d = self.grad_reducer_d(grads_d)
        return ops.depend(loss_w, self.optimizer_w(grads_w)), ops.depend(
            loss_d, self.optimizer_d(grads_d)
        )


class PredictWithSigmoid(nn.Cell):
    """
    Predict definition
    """

    def __init__(self, network):
        super().__init__()
        self.network = network
        self.sigmoid = ops.Sigmoid()
        parallel_mode = context.get_auto_parallel_context("parallel_mode")
        full_batch = context.get_auto_parallel_context("full_batch")
        is_auto_parallel = parallel_mode in (
            ParallelMode.SEMI_AUTO_PARALLEL,
            ParallelMode.AUTO_PARALLEL,
        )
        if is_auto_parallel and full_batch:
            self.sigmoid.shard(((1, 1),))

    def construct(self, batch_ids, batch_wts, labels):
        (
            logits,
            _,
        ) = self.network(batch_ids, batch_wts)
        pred_probs = self.sigmoid(logits)
        return logits, pred_probs, labels

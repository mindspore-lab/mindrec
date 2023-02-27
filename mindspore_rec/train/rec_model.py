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
from mindspore import Model
from mindspore import nn
from mindspore import context
from mindspore.parallel._utils import _device_number_check
from mindspore._checkparam import Validator
from mindspore.train.callback import _InternalCallbackParam, RunContext, _CallbackManager
from mindspore import log as logger

class RecModel(Model):
    """
    A high-level API for recommending network models, providing interfaces such as online training.

    Args:
        network (Cell): A training network.
        loss_fn (Cell): Objective function. If `loss_fn` is None, the `network` should contain the calculation of loss
                        and parallel if needed. Default: None.
        optimizer (Cell): Optimizer for updating the weights. If `optimizer` is None, the `network` needs to
                          do backpropagation and update weights. Default value: None.
        metrics (Union[dict, set]): A Dictionary or a set of metrics for model evaluation.
                                    eg: {'accuracy', 'recall'}. Default: None.
        eval_network (Cell): Network for evaluation. If not defined, `network` and `loss_fn` would be wrapped as
                             `eval_network` . Default: None.
        eval_indexes (list): It is used when eval_network is defined. If `eval_indexes` is None by default, all outputs
                             of the `eval_network` would be passed to metrics. If `eval_indexes` is set, it must contain
                             three elements: the positions of loss value, predicted value and label in outputs of the
                             `eval_network`. In this case, the loss value will be passed to the `Loss` metric, the
                             predicted value and label will be passed to other metrics.
                             `mindspore.train.Metric.set_indexes
                             <https://www.mindspore.cn/docs/en/master/api_python/train/mindspore.train.Metric.html?#mindspore.train.Metric.set_indexes>`_
                             is recommended instead of `eval_indexes`.
                             Default: None.
        amp_level (str): Option for argument `level` in `mindspore.amp.build_train_network
            <https://www.mindspore.cn/docs/en/master/api_python/amp/mindspore.amp.build_train_network.html>`_,
            level for mixed precision training. Supports ["O0", "O2", "O3", "auto"]. Default: "O0".

            - "O0": Do not change.
            - "O2": Cast network to float16, keep BatchNorm run in float32, using dynamic loss scale.
            - "O3": Cast network to float16, the BatchNorm is also cast to float16, loss scale will not be used.
            - auto: Set level to recommended level in different devices. Set level to "O2" on GPU, set
              level to "O3" on Ascend. The recommended level is chosen by the expert experience, not applicable to all
              scenarios. User should specify the level for special network.

            "O2" is recommended on GPU, "O3" is recommended on Ascend.
            The BatchNorm strategy can be changed by `keep_batchnorm_fp32` settings in `kwargs`. `keep_batchnorm_fp32`
            must be a bool. The loss scale strategy can be changed by `loss_scale_manager` setting in `kwargs`.
            `loss_scale_manager` should be a subclass of `mindspore.amp.LossScaleManager
            <https://www.mindspore.cn/docs/en/master/api_python/amp/mindspore.amp.LossScaleManager.html>`_.
            The more detailed explanation of `amp_level` setting can be found at `mindspore.amp.build_train_network
            <https://www.mindspore.cn/docs/en/master/api_python/amp/mindspore.amp.build_train_network.html>`_.

        boost_level (str): Option for argument `level` in `mindspore.boost`, level for boost mode
            training. Supports ["O0", "O1", "O2"]. Default: "O0".

            - "O0": Do not change.
            - "O1": Cast the operators in white_list to float16, the remaining operators are kept in float32.
            - "O1": Enable the boost mode, the performance is improved by about 20%, and
              the accuracy is the same as the original accuracy.
            - "O2": Enable the boost mode, the performance is improved by about 30%, and
              the accuracy is reduced by less than 3%.

            If you want to config boost mode by yourself, you can set boost_config_dict as `boost.py`.
            In order for this function to work, you need to set the optimizer, eval_network or metric parameters
            at the same time.

            Notice: The current optimization enabled by default only applies to some networks, and not all networks
            can obtain the same benefits.  It is recommended to enable this function on
            the Graph mode + Ascend platform, and for better acceleration, refer to the documentation to configure
            boost_config_dict.
    """

    def __init__(self, network, loss_fn=None, optimizer=None, metrics=None, eval_network=None,
                 eval_indexes=None, amp_level="O0", boost_level="O0"):
        super(RecModel, self).__init__(network=network, loss_fn=loss_fn, optimizer=optimizer, metrics=metrics,
                                       eval_network=eval_network, eval_indexes=eval_indexes, amp_level=amp_level,
                                       boost_level=boost_level)


    def online_train(self, train_dataset, callbacks=None, dataset_sink_mode=True, sink_size=1):
        """
        Online training API for recommend model.

        Note:
            If dataset_sink_mode is True, data will be sent to device queue. If the device is Ascend, features
            of data will be transferred one by one. The limitation of data transmission per time is 256M.

            When dataset_sink_mode is True, the `step_end` method of the instance of Callback will be called at the end
            of epoch, and dataset will be bound to this model and cannot be used by other models.

            When setting device target is CPU, the training process will be performed with dataset not sink mode.

            The dataset for online training is unbounded, it has an infinite number of batch data,
            which is the main difference from the offline training dataset.

        Args:
            train_dataset (Dataset): A training dataset iterator. If `loss_fn` is defined, the data and label will be
                                     passed to the `network` and the `loss_fn` respectively, so a tuple (data, label)
                                     should be returned from dataset. If there is multiple data or labels, set `loss_fn`
                                     to None and implement calculation of loss in `network`,
                                     then a tuple (data1, data2, data3, ...) with all data returned from dataset will be
                                     passed to the `network`.
                                     The train dataset is unbounded.
            callbacks (Optional[list[Callback], Callback]): List of callback objects or callback object,
                                                            which should be executed while training.
                                                            Default: None.
            dataset_sink_mode (bool): Determines whether to pass the data through dataset channel.
                                      Configure device target of CPU, the training process will be performed with
                                      dataset not sink. Default: True.
            sink_size (int): Controls how many batches of data in each sink. Default: 1
        """
        Validator.check_bool(dataset_sink_mode)
        if isinstance(self._train_network, nn.GraphCell) and dataset_sink_mode:
            raise ValueError("Dataset sink mode is currently not supported when training with a GraphCell.")

        _device_number_check(self._parallel_mode, self._device_number)

        if callbacks:
            self._check_methods_for_custom_callbacks(callbacks, "train")

        if self._parameter_broadcast:
            self._train_network.set_broadcast_flag()

        cb_params = _InternalCallbackParam()
        cb_params.train_network = self._train_network
        if dataset_sink_mode:
            cb_params.batch_num = sink_size
        else:
            cb_params.batch_num = train_dataset.get_dataset_size()

        with _CallbackManager(callbacks) as list_callback:
            self._check_reuse_dataset(train_dataset)
            if not dataset_sink_mode:
                self._online_train_dataset_not_sink(train_dataset, list_callback, cb_params)
            elif context.get_context("device_target") == "CPU":
                logger.info("The CPU doesn't support dataset sink mode currently,"
                            "so the training process will be performed with dataset not sink.")
                self._online_train_dataset_not_sink(train_dataset, list_callback, cb_params)
            else:
                self._online_train_dataset_sink(train_dataset, list_callback, cb_params, sink_size)


    def _online_train_dataset_not_sink(self, train_dataset, callbacks=None, cb_params=None):
        """
        Training process for feed mode. The train input data would be passed to network directly.

        Args:
            train_dataset (Dataset): A training dataset iterator. If there is no
                                     loss_fn, a tuple with multiple data (data1, data2, data3, ...) should be
                                     returned and passed to the network. Otherwise, a tuple (data, label) should
                                     be returned. The data and label would be passed to the network and loss
                                     function respectively.
            callbacks (Callback): Executor of callback list. Default: None.
            cb_params (_InternalCallbackParam): Callback parameters. Default: None.
        """
        dataset_helper, _ = self._exec_preprocess(is_train=True,
                                                  dataset=train_dataset,
                                                  dataset_sink_mode=False,
                                                  epoch_num=-1)

        cb_params.cur_epoch_num = 0
        cb_params.cur_step_num = 0
        cb_params.dataset_sink_mode = False
        run_context = RunContext(cb_params)
        callbacks.on_train_begin(run_context)

        max_epoch = sys.maxsize
        # Epoch iteration
        for epoch_iter in range(max_epoch):
            cb_params.cur_epoch_num = epoch_iter + 1
            # Epoch callback begin
            callbacks.on_train_epoch_begin(run_context)

            # Step iteration
            for next_element in dataset_helper:
                cb_params.cur_step_num += 1
                # Step callback begin
                callbacks.on_train_step_begin(run_context)
                self._check_network_mode(self._train_network, True)
                outputs = self._train_network(*next_element)
                cb_params.net_outputs = outputs

                # Handle loss scale.
                if self._loss_scale_manager and self._loss_scale_manager.get_drop_overflow_update():
                    _, overflow, _ = outputs
                    overflow = np.all(overflow.asnumpy())
                    self._loss_scale_manager.update_loss_scale(overflow)

                # Step callback end
                callbacks.on_train_step_end(run_context)

            train_dataset.reset()
            # Epoch callback end
            callbacks.on_train_epoch_end(run_context)

        callbacks.on_train_end(run_context)

    def _online_train_dataset_sink(self, train_dataset, callbacks=None, cb_params=None, sink_size=1):
        """
        Training process for data sink mode. The data would be passed to network through dataset channel.

        Args:
            train_dataset (Dataset): A training dataset iterator. If there is no
                                     loss_fn, a tuple with multiple data (data1, data2, data3, ...) should be
                                     returned and passed to the network. Otherwise, a tuple (data, label) should
                                     be returned. The data and label would be passed to the network and loss
                                     function respectively.
            callbacks (Callback): Executor of callback list. Default: None.
            cb_params (_InternalCallbackParam): Callback parameters. Default: None.
            sink_size (int): Controls how many batches of data in each sink. Default: 1
        """
        sink_size = Validator.check_positive_int(sink_size)
        if sink_size != 1:
            raise ValueError(f"The sink_size parameter only support value of 1 currently, but got: {sink_size}")

        cb_params.cur_step_num = 0
        cb_params.dataset_sink_mode = True
        run_context = RunContext(cb_params)

        callbacks.on_train_begin(run_context)
        dataset_helper = None
        if hasattr(train_dataset, '_dataset_helper'):
            dataset_helper = train_dataset._dataset_helper
    
        max_epoch = sys.maxsize
        # Epoch iteration
        for epoch_iter in range(max_epoch):
            cb_params.cur_epoch_num = epoch_iter + 1 

            callbacks.on_train_epoch_begin(run_context)
            dataset_helper, train_network = self._exec_preprocess(is_train=True,
                                                                  dataset=train_dataset,
                                                                  dataset_sink_mode=True,
                                                                  sink_size=sink_size,
                                                                  epoch_num=-1,
                                                                  dataset_helper=dataset_helper)
            cb_params.train_network = train_network

            # Train sink_size batchs once.
            for inputs in dataset_helper:
                cb_params.cur_step_num += sink_size
                callbacks.on_train_step_begin(run_context)
                train_network = self._check_network_mode(train_network, True)
                outputs = train_network(*inputs)
                cb_params.net_outputs = outputs
                callbacks.on_train_step_end(run_context)
    
            callbacks.on_train_epoch_end(run_context)
        callbacks.on_train_end(run_context)

mindspore_rec
========================

MindSpore推荐领域深度模型的训练加速库。

.. py:class:: mindspore_rec.RecModel(network, loss_fn=None, optimizer=None, metrics=None, eval_network=None, eval_indexes=None, amp_level="O0", boost_level="O0")

    推荐模型训练的高层API封装，提供了在线训练等功能接口。

    参数：
        - **network** (Cell) - 用于训练或推理的神经网络。
        - **loss_fn** (Cell) - 损失函数。如果 `loss_fn` 为None，`network` 中需要进行损失函数计算，必要时也需要进行并行计算。默认值：None。
        - **optimizer** (Cell) - 用于更新网络权重的优化器。如果 `optimizer` 为None， `network` 中需要进行反向传播和网络权重更新。默认值：None。
        - **metrics** (Union[dict, set]) - 用于模型评估的一组评价函数。例如：{'accuracy', 'recall'}。默认值：None。
        - **eval_network** (Cell) - 用于评估的神经网络。未定义情况下，`Model` 会使用 `network` 和 `loss_fn` 封装一个 `eval_network` 。默认值：None。
        - **eval_indexes** (list) - 在定义 `eval_network` 的情况下使用。如果 `eval_indexes` 为默认值None，`Model` 会将 `eval_network` 的所有输出传给 `metrics` 。如果配置 `eval_indexes` ，必须包含三个元素，分别为损失值、预测值和标签在 `eval_network` 输出中的位置，此时，损失值将传给损失评价函数，预测值和标签将传给其他评价函数。推荐使用评价函数的 `mindspore.train.Metric.set_indexes <https://www.mindspore.cn/docs/zh-CN/master/api_python/train/mindspore.train.Metric.html?#mindspore.train.Metric.set_indexes>`_ 代替 `eval_indexes` 。默认值：None。
        - **amp_level** (str) - `mindspore.amp.build_train_network <https://www.mindspore.cn/docs/zh-CN/master/api_python/amp/mindspore.amp.build_train_network.html>`_ 的可选参数 `level` ， `level` 为混合精度等级，该参数支持["O0", "O2", "O3", "auto"]。默认值："O0"。

          - "O0": 不变化。
          - "O2": 将网络精度转为float16，BatchNorm保持float32精度，使用动态调整损失缩放系数（loss scale）的策略。
          - "O3": 将网络精度（包括BatchNorm）转为float16，不使用损失缩放策略。
          - auto: 为不同处理器设置专家推荐的混合精度等级，如在GPU上设为"O2"，在Ascend上设为"O3"。该设置方式可能在部分场景下不适用，建议用户根据具体的网络模型自定义设置 `amp_level` 。

          在GPU上建议使用"O2"，在Ascend上建议使用"O3"。
          关于 `amp_level` 详见 `mindpore.amp.build_train_network <https://www.mindspore.cn/docs/zh-CN/master/api_python/amp/mindspore.amp.build_train_network.html>`_。

        - **boost_level** (str) - `mindspore.boost` 的可选参数，为boost模式训练等级。支持["O0", "O1", "O2"]. 默认值："O0"。

          - "O0": 不变化。
          - "O1": 启用boost模式，性能将提升约20%，准确率保持不变。
          - "O2": 启用boost模式，性能将提升约30%，准确率下降小于3%。

          如果想自行配置boost模式，可以将 `boost_config_dict` 设置为 `boost.py`。
          为使功能生效，需要同时设置optimizer、eval_network或metric参数。
          注意：当前默认开启的优化仅适用部分网络，并非所有网络都能获得相同收益。建议在图模式+Ascend平台下开启该模式，同时为了获取更好的加速效果，请参考文档配置boost_config_dict。

    .. py:method:: online_train(train_dataset, callbacks=None, dataset_sink_mode=True, sink_size=1)

        启动在线训练。

        .. note::
            - 如果 `dataset_sink_mode` 配置为True，数据将被送到处理器中。如果处理器是Ascend，数据特征将被逐一传输，每次数据传输的上限是256M。
            - 如果 `dataset_sink_mode` 配置为True，仅在每个epoch结束时调用Callback实例的step_end方法。
            - 如果计算设备指定为CPU，则只支持非数据下沉模式。
            - 与离线数据集相比，在线数据集是无界数据流，训练过程会持续进行。

        参数：
            - **train_dataset** (Dataset) - 一个用于在线训练的无界训练数据集迭代器。一个训练数据集迭代器。如果定义了 loss_fn ，则数据和标签会被分别传给 network 和 loss_fn ，此时数据集需要返回一个元组（data, label）。如果数据集中有多个数据或者标签，可以设置 loss_fn 为None，并在 network 中实现损失函数计算，此时数据集返回的所有数据组成的元组（data1, data2, data3, …）会传给network。
            - **callbacks** (Optional[list[Callback], Callback]) - 训练过程中需要执行的回调对象或者回调对象列表。默认值：None。
            - **dataset_sink_mode** (bool) - 数据是否直接下沉至处理器进行处理。使用CPU处理器时，模型训练流程将以非下沉模式执行。默认值：True。
            - **sink_size** (int) - 控制每次数据下沉的数据量。dataset_sink_mode 为False时 sink_size 无效。默认值：1。
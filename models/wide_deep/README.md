# 目录
- [目录](#目录)
- [Wide&Deep概述](#widedeep概述)
- [数据集](#数据集)
- [环境要求](#环境要求)
- [快速开始](#快速开始)
- [极致性能体验](#极致性能体验)
- [随机情况说明](#随机情况说明)


# Wide&Deep概述

Wide&Deep模型是推荐和点击预测领域的经典模型。  [Wide&Deep推荐系统学习](https://arxiv.org/pdf/1606.07792.pdf)论文中描述了如何实现Wide&Deep。
Wide&Deep模型训练了宽线性模型和深度学习神经网络，结合了推荐系统的记忆和泛化的优点。

# 数据集

- 详见recommender/datasets目录

# 环境要求

- 硬件（Ascend或GPU）
    - 准备Ascend或GPU处理器搭建硬件环境。
- 框架
    - [MindSpore](https://gitee.com/mindspore/mindspore)
- 如需查看详情，请参见如下资源：
    - [MindSpore教程](https://www.mindspore.cn/tutorials/zh-CN/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/docs/api/zh-CN/master/index.html)


# 快速开始

**1. 克隆代码**

```bash
git clone https://gitee.com/mindspore/recommender.git
cd recommender/models/wide_deep
```

目录结构如下：

```bash
└── wide_deep
    ├── default_config.yaml
    ├── eval.py
    ├── README.md
    ├── requirements.txt
    ├── scripts
    │   ├── run_distribute_train_for_ascend.sh
    │   ├── run_distribute_train_for_gpu.sh
    │   ├── run_parameter_server_distribute_train.sh
    │   ├── run_parameter_server_standalone_train.sh
    │   ├── run_standalone_train_for_ascend.sh
    │   └── run_standalone_train_for_gpu.sh
    ├── src
    │   ├── callbacks.py
    │   ├── datasets.py
    │   ├── __init__.py
    │   ├── metrics.py
    │   ├── model_utils
    │   │   ├── config.py
    │   │   ├── device_adapter.py
    │   │   ├── __init__.py
    │   │   └── local_adapter.py
    │   └── wide_and_deep.py
    ├── train_and_eval_distribute.py
    ├── train_and_eval_parameter_server_distribute.py
    ├── train_and_eval_parameter_server_standalone.py
    ├── train_and_eval.py
    └── train.py
```

**2. 下载数据集并进行格式转换**

详见datasets目录，使用对应脚本预处理数据，生成的MindRecord数据并存放在./data/mindrecord下。

**3. 开始训练**

- 训练脚本参数

``train.py``、``train_and_eval.py``、``train_and_eval_distribute.py``、``train_and_eval_parameter_server_standalone.py``和``train_and_eval_parameter_server_distribute.py``的参数设置可通过配置文件default_config.yaml来修改，详见default_config.yaml的Config description。


- **单卡模式**

**GPU环境：**

```bash
python train_and_eval.py --data_path=./data/mindrecord --device_target="GPU"
```


**Ascend环境：**

```bash
python train_and_eval.py --data_path=./data/mindrecord --device_target="Ascend"
```


- **多卡数据并行模式**

**GPU 8卡：**

GPU多卡数据并行可采用动态组网方式，详细见连接：https://www.mindspore.cn/tutorials/experts/zh-CN/r1.8/parallel/train_gpu.html?highlight=openmpi，中非openmpi启动方式, 需要指定Scheduler的ip和port、Worker的数量。
```bash
bash scripts/run_distribute_train_for_gpu.sh RANK_SIZE EPOCHS DATASET SCHED_HOST SCHED_PORT
```
如
```bash
bash scripts/run_distribute_train_for_gpu.sh 8 5 ./data/mindrecord 127.0.0.1 2898
```
表示8 Worker训练5个epoch，Scheduler的ip和port分别是127.0.0.1、2898, 训练日志保存在./worker_*/内。


**Ascend 8卡：**

```bash
bash scripts/run_distribute_train_for_ascend.sh RANK_SIZE EPOCHS DATASET RANK_TABLE_FILE
```
如
```bash
bash scripts/run_distribute_train_for_ascend.sh 8 5 ./data/mindrecord ${rank_table_file_path}
```
表示8 Worker训练5个epoch， 训练日志保存在./device_*/内。


- **Embedding Cache模式**

Embedding Cache模式基于Parameter Server架构，多卡（多Worker）训练时需要开启自动并行（context.set_auto_parallel_context(parallel_mode=ParallelMode.AUTO_PARALLEL）），配置Embedding Table为行切并配置Embedding Cache大小（EmbeddingLookup(slice_mode=nn.EmbeddingLookup.TABLE_ROW_SLICE, vocab_cache_size=vocab_cache_size)，vocab_cache_size入参表示在Worker侧device上缓存的Embedding Table的行数），设置Scheduler的ip和port、Worker的数量，server的数量。Parameter Server和Embedding Cache详细介绍参考：https://www.mindspore.cn/tutorials/experts/zh-CN/r1.8/parallel/parameter_server_training.html


**GPU 单卡：**

```bash
bash scripts/run_parameter_server_standalone_train.sh EPOCHS DEVICE_TARGET DATASET SERVER_NUM SCHED_HOST SCHED_PORT VOCAB_CACHE_SIZE [DEVICE_ID]
# DEVICE_ID是可选的，表示使用的GPU卡号，默认值为0
```
如
```bash
bash scripts/run_parameter_server_standalone_train.sh 5 "GPU" ./data/mindrecord 1 127.0.0.1 2898 5860000
```
表示在GPU环境上执行5个epoch训练，Embedding Cache Size为5860000（即Embedding Table总大小，配成和总表大小相同，训练一定step后，即可达到100%命中cache），1个Server，Scheduler的ip和port分别是127.0.0.1、2898, 训练日志保存在./worker/内。

Ascend环境只需要将DEVICE_TARGET参数指定为"Ascend"即可。

如果设备host内存和磁盘空间足够，该模式可以实现单Ascend/GPU训练TB级别推荐模型, 可以直接使用scripts/run_parameter_server_standalone_train_terabyte_scale_model.sh脚本进行训练：
```bash
bash scripts/run_parameter_server_standalone_train_terabyte_scale_model.sh EPOCHS DEVICE_TARGET DATASET SERVER_NUM SCHED_HOST SCHED_PORT VOCAB_CACHE_SIZE [DEVICE_ID]
```
训练大模型需要导出环境变量`MS_EMBEDDING_REMOTE_CACHE_MEMORY_SIZE`，来约束remote(server)侧最大允许使用的内存大小，超出部分自动落盘。如最大允许使用10GB主存，可如下导出环境变量：
```bash
export MS_EMBEDDING_REMOTE_CACHE_MEMORY_SIZE=10
```
run_parameter_server_standalone_train_terabyte_scale_model.sh脚本中默认设置的MS_EMBEDDING_REMOTE_CACHE_MEMORY_SIZE值为10GB，可以根据环境配置进行调整，该值越大，训练性能越好。

**GPU 8卡**：

```bash
bash scripts/run_parameter_server_distribute_train.sh RANK_SIZE EPOCHS DEVICE_TARGET DATASET SERVER_NUM SCHED_HOST SCHED_PORT VOCAB_CACHE_SIZE
```
如：
```bash
bash scripts/run_parameter_server_distribute_train.sh 8 5 "GPU" ./data/mindrecord 1 127.0.0.1 2898 1000000
```
表示在GPU 8卡环境上执行5个epoch训练，Embedding Cache Size为1000000，1个Server，Scheduler的ip和port分别是127.0.0.1、2898, 训练日志保存在./worker_*/内。


**Ascend 8卡：**

```bash
bash scripts/run_parameter_server_distribute_train.sh RANK_SIZE EPOCHS DEVICE_TARGET DATASET SERVER_NUM SCHED_HOST SCHED_PORT VOCAB_CACHE_SIZE RANK_TABLE_FILE
```
如：
```bash
bash scripts/run_parameter_server_distribute_train.sh 8 5 "Ascend" ./data/mindrecord 1 127.0.0.1 2898 1000000 ${rank_table_file_path}

```
表示在Ascend 8卡环境上执行5个epoch训练，Embedding Cache Size为1000000，1个Server，Scheduler的ip和port分别是127.0.0.1、2898, ${rank_table_file_path}表示环境中rank table file的绝对路径, 训练日志保存在./worker_*/内。


# 极致性能体验

MindSpore从1.1.1版本之后，支持通过开启numa亲和获得极致的性能，需要安装numa库：

- ubuntu : sudo apt-get install libnuma-dev
- centos/euleros : sudo yum install numactl-devel

1.1.1版本支持设置config的方式开启numa亲和：

import mindspore.dataset as de
de.config.set_numa_enable(True)

1.2.0版本进一步支持了环境变量开启numa亲和：

export DATASET_ENABLE_NUMA=True


# 随机情况说明

以下三种随机情况：

- 数据集的打乱。
- 模型权重的随机初始化。
- dropout算子。

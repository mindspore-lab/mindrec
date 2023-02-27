# Online Learning

## 概述

推荐网络模型更新的实时性要求较高，online learning的方式可有效特性推荐网络模型更新实时性， 提高模型精度与点击通过率。

在线训练与离线训练主要区别：

1. 在线学习训练数据为流式数据、无确定的dataset size、epoch，离线训练训练数据有确定的data set size、epoch。
2. 在线学习为服务，持续训练，离线训练训练完离线数据集后退出。
3. 在线训练需要收集并存储训练数据，收集到固定数量的数据或者经过一定时间窗口后驱动训练流程。



## 整体架构

用户的流式训练数据推送的 Kafka 中，MindPandas 从 Kafka 读取数据并进行特征工程转换，然后写入分布式计算引擎中，MindData 从分布式计算引擎中读取数据作为训练数据进行训练，MindSpore进程作为服务常驻，有数据接入就进行训练，并定期导出 ckpt，整体流程见下图1。

图1  在线训推一体化 E2E部署视图

![image.png](https://foruda.gitee.com/images/1665653730199149252/d77df81a_7356746.png)



## 新增API

```python
RecModel.online_train(self, train_dataset, callbacks=None, dataset_sink_mode=True, sink_size=1)
```

参数：

-  train_dataset(Dataset) - 在线训练数据集，包含训练数据和label，该数据集无边界，dataset_size == sys.maxsize。默认值：None。
-  callbacks(Optional[list[Callback], Callback]) - 训练过程中执行的callbacks。默认值：None 。
-  dataset_sink_mode(bool) - 是否开启数据下沉，如果开启，数据将通过dataset channel发送到device queue中。默认值： True。
-  sink_size ( int) - 控制一次下沉多少个batch的数据。默认值： 1。


使用前先安装mindspore_rec推荐套件，安装方式见[ReadMe](../../README.md)。

example：
```
from mindspore_rec import RecModel as Model
#model定义同mindspore.model
...
model.online_train(self, train_dataset, callbacks=None, dataset_sink_mode=True)

```



## 使用约束

- online learning数据模块依赖MindPandas，MindPandas当前支持Python版本为3.8，所以online learning需要使用3.8版本的Python，对应于MindSpore及recommender套件都使用Python3.8版本。

- 目前支持 GPU后端、静态图模式、Linux平台
- 暂不支持fullbatch
- 训练侧新增接口（位于recommender仓），sink_size入参目前仅支持1，默认值也为1:

```
RecModel.online_train(self, train_dataset, callbacks=None, dataset_sink_mode=True, sink_size=1)
```



## Python包依赖

mindpandas  v0.1.0

mindspore_rec  v0.2.0

kafka-python v2.0.2



## 使用样例

下面以Criteo数据集训练Wide&Deep为例，介绍一下整个online learning流程，样例代码位于[examples/online_learning](../../examples/online_learning)

### 下载Kafka
```shell
wget https://archive.apache.org/dist/kafka/3.2.0/kafka_2.13-3.2.0.tgz
tar -xzf kafka_2.13-3.2.0.tgz
```

如需安装其他版本，请参照https://archive.apache.org/dist/kafka/

### 启动kafka-zookeeper

```shell
cd kafka_2.13-3.2.0
bin/zookeeper-server-start.sh config/zookeeper.properties
```

### 启动kafka-server
打开另一个命令终端，启动kafka服务

```shell
cd kafka_2.13-3.2.0
bin/kafka-server-start.sh config/server.properties
```

### 启动kafka_client

进入recommender仓在线学习样例目录，启动kafka_client

```shell
cd recommender/examples/online_learning
python kafka_client.py
```

kafka_client只需要启动一次，可以使用Kafka设置topic对应的partition数量

### 启动分布式计算引擎

   ```bash
   yrctl start --master  --address $MASTER_HOST_IP  
   
   # 参数说明
   # --master： 表示当前host为master节点，非master节点不用指定‘--master’参数
   # --address： master节点的ip
   ```

### 启动数据producer

   producer 用于模拟在线训练场景，将本地的criteo数据集(数据集下载参考[dataset下载](../../datasets/criteo_1tb/))写入到Kafka，供consumer使用

   当前样例使用多进程读取两个文件，并将数据写入Kafka

   ```bash
   python producer.py
   
   # 参数说明
   # --file1： criteo数据集在本地磁盘的存放路径
   # --file2： criteo数据集在本地磁盘的存放路径
   # 上述文件均为criteo原始数据集文本文件，file1和file2可以被并发处理，file1和file2可以相同也可以不同，如果相同则相当于文件中每个样本被使用两次。
   ```

### 启动数据consumer

   consumer既是Kafka消费者，同时也是sender，将处理好的data frame放入MindPandas.channel中，channel详情请参考:[MindPandas](https://gitee.com/mindspore/mindpandas)，
   consumer为criteo数据集进行特征工程需要3个数据集相关文件: `all_val_max_dict.pkl`、 `all_val_min_dict.pkl`和`cat2id_dict.pkl`。`$PATH_TO_VAL_MAX_DICT`、 `$PATH_TO_VAL_MIN_DICT`和`$PATH_TO_CAT_TO_ID_DICT` 分别为这些文件在环境上的绝对路径。这3个pkl文件具体生产方法可以参考[process_data.py](../../datasets/criteo_1tb/process_data.py)，对原始criteo数据集做转换生产对应的.pkl文件。

   ```bash
   python consumer.py  --num_shards=$DEVICE_NUM  --address=$LOCAL_HOST_IP  --max_dict=$PATH_TO_VAL_MAX_DICT  --min_dict=$PATH_TO_VAL_MIN_DICT  --map_dict=$PATH_TO_CAT_TO_ID_DICT
   ```
文件路径参数:

- max_dict - 稠密特征列的最大值特征文件。默认值：'./all_val_max_dict.pkl'。
- min_dict - 稠密特征值的最小值特征文件。默认值：'./all_val_min_dict.pkl'。
- max_dict - 稀疏特征列的字典文件。默认值：'./cat2id_dict.pkl'。

MindPandas channel.DataSender相关参数：

- num_shards(int)  - 指定数据的切片数量，对应训练侧的device卡数，单卡则设置为1，8卡设置为8。默认值：1。
- address (str) - 当前sender运行节点的ip。默认值：127.0.0.1。
- namespace (str) - channel所属的命名空间，不同命名空间的DataSender和DataReceiver不能互连。默认值：  ’demo‘。
- dataset_name(str) - 数据集名称。默认值：’criteo‘。

#### MindPandas channel.sender使用示例：

   ```bash
   import mindpandas as pd
   from mindpandas.channel import DataSender
   import numpy as np
   import time
   
   if __name__ == '__main__':
       # 初始化sender，建立channel，数据集设为'dataset', 设置将数据集划分为5个分片
       sender = DataSender(address='127.0.0.1', dataset_name='dataset', num_shards=5)
       while True:
           # 调用send接口向channel中发送数据
           sender.send(df)
           print("========Data Sender========")
           time.sleep(10)
   ```

### 启动训练
####  训练侧通过MindPandas.channel.receiver读取数据，使用样例如下：

   ```bash
   from mindpandas.channel import DataReceiver
   
   if __name__ == '__main__':
       # 初始化receiver，连接channel，数据集名为'dataset', 指定获取shard_id为0的分片
       receiver = DataReceiver(address='127.0.0.1', dataset_name='dataset', shard_id=0)
       while True:
           # 调用receiver接口从channel中读取数据
           data = receiver.recv()
           print("========Data Sender========")
           print(data)
   ```

   config采用yaml的形式，见[default_config.yaml](../../examples/online_learning/default_config.yaml)

   单卡训练：

   ```bash
   python online_train.py --address=$LOCAL_HOST_IP   --dataset_name=criteo 
   
   # 参数说明：
   # --address： 本机host ip，从MindPandas接收训练数据需要配置
   # --dataset_name： 数据集名字，和consumer模块保持一致
   ```

   多卡训练MPI方式启动：

   ```bash
   bash mpirun_dist_online_train.sh [$RANK_SIZE] [$LOCAL_HOST_IP]
   
   # 参数说明：
   # RANK_SIZE：多卡训练卡数量
   # LOCAL_HOST_IP：本机host ip，用于MindPandas接收训练数据
   ```


   动态组网方式启动多卡训练：

   ```bash
   bash run_dist_online_train.sh [$WORKER_NUM] [$SHED_HOST] [$SCHED_PORT] [$LOCAL_HOST_IP]
   
   # 参数说明：
   # WORKER_NUM：多卡训练卡数量
   # SHED_HOST：MindSpore动态组网需要的Scheduler 角色的IP
   # SCHED_PORT：MindSpore动态组网需要的Scheduler 角色的Port
   # LOCAL_HOST_IP：本机host ip，从MindPandas接收训练数据需要配置
   ```

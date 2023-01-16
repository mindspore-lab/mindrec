# 目录

<!-- TOC -->

- [目录](#目录)
- [Deep&Cross描述](#deepcross描述)
- [模型架构](#模型架构)
- [数据集](#数据集)
- [环境要求](#环境要求)
- [快速入门](#快速入门)
- [脚本说明](#脚本说明)
    - [脚本及样例代码](#脚本及样例代码)
    - [脚本参数](#脚本参数)
    - [训练过程](#训练过程)
        - [训练](#训练)
        - [分布式训练](#分布式训练)

<!-- /TOC -->

# Deep&Cross描述

Deep & Cross Network(DCN)是来自于 2017 年 google 和 Stanford 共同完成的一篇工作，用于广告场景下的点击率预估（CTR），对比同样来自 google 的工作 Wide & Deep，DCN 不需要特征工程来获得高阶的交叉特征，对比 FM 系列的模型，DCN 拥有更高的计算效率并且能够提取到更高阶的交叉特征。

[论文](https://arxiv.org/pdf/1708.05123.pdf)

# 模型架构

DCN模型最开始是Embedding and stacking layer，然后是并行的Cross Network和Deep Network，最后是Combination Layer把Cross Network和Deep Network的结果组合得到输出。

# 数据集

使用的数据集：criteo 1tb click logs

# 环境要求

- 硬件（GPU）
    - 使用GPU处理器来搭建硬件环境。
- 框架
    - [MindSpore](https://www.mindspore.cn/install)

# 快速入门

通过官方网站安装MindSpore后，您可以按照如下步骤进行训练和评估：

1. 克隆代码。

```bash
git clone https://gitee.com/mindspore/recommender.git
cd recommender/rec/models/deep_and_cross
```

2. 下载数据集。

  > 请参考recommender/datasets/criteo_1tb。

```bash
sh download.sh 1
```

3. 使用此脚本预处理数据。

  > 请参考recommender/datasets/criteo_1tb。

处理过程可能需要30min，生成的MindRecord数据存放在~/mindspore/rec/criteo_1tb_data/mindrecord路径下。

```bash
process_data.py --data_path=~/mindspore/rec/criteo_1tb_data --part_num=1
```

4. 开始训练。

数据集准备就绪后，即可在GPU上训练和评估模型。

GPU单卡训练命令如下：

```bash
#单卡训练示例
python train.py --device_target="GPU"  --data_path="~/mindspore/rec/criteo_1tb_data/mindrecord" > output.train.log 2>&1 &
#或
bash scripts/run_train_gpu.sh "~/mindspore/rec/criteo_1tb_data/mindrecord"
```

GPU 8卡训练命令如下：

```bash
#8卡训练示例
bash scripts/run_train_multi_gpu.sh "~/mindspore/rec/criteo_1tb_data/mindrecord"
```

5. 开始验证。

训练完毕后，按如下操作评估模型。

```bash
python eval.py --ckpt_path=CHECKPOINT_PATH
#或
bash scripts/run_eval.sh CHECKPOINT_PATH
```

# 脚本说明

## 脚本及样例代码

```bash
├── model_zoo
    ├── README.md                           // 所有模型相关说明
    ├── deep_and_cross
        ├── README.md                       // deep and cross相关说明
        ├── scripts
        │   ├──run_train_gpu.sh             // GPU处理器单卡训练shell脚本
        │   ├──run_train_multi_gpu.sh       // GPU处理器8卡训练shell脚本
        │   ├──run_eval.sh                  // 评估的shell脚本
        ├── src
        │   ├──dataset.py                   // 创建数据集
        │   ├──deepandcross.py              // deepandcross架构
        │   ├──callback.py                  // 定义回调
        │   ├──config.py                    // 参数配置
        │   ├──metrics.py                   // 定义AUC
        │   ├──preprocess_data.py           // 预处理数据，生成mindrecord文件
        ├── train.py                        // 训练脚本
        ├── eval.py                         // 评估脚本
```

## 脚本参数

在config.py中可以同时配置训练参数和评估参数。


  ```python
  self.device_target = "GPU"                     #设备选择
  self.device_id = 0                             #用于训练或评估数据集的设备ID
  self.epochs = 10                               #训练轮数
  self.batch_size = 16000                        #batch size大小
  self.deep_layer_dim = [1024, 1024]             #deep and cross deeplayer层大小
  self.cross_layer_num = 6                       #deep and cross crosslayer层数
  self.eval_file_name = "eval.log"               #验证结果输出文件
  self.loss_file_name = "loss.log"               #loss结果输出文件
  self.ckpt_path = "./checkpoints/"              #checkpoints输出目录
  self.dataset_type = "mindrecord"               #数据格式
  self.is_distributed = 0                        #是否分布式训练

  ```

更多配置细节请参考脚本`config.py`。

## 训练过程

### 训练

- GPU处理器环境运行

  ```bash
  sh scripts/run_train_gpu.sh
  ```

  上述bash命令将在后台运行，您可以通过output.train.log文件查看结果。

  训练结束后，您可在默认`./checkpoints/`脚本文件夹下找到检查点文件。

### 分布式训练

- GPU处理器环境运行

  ```bash
  bash scripts/run_train_multi_gpu.sh
  ```

  上述shell脚本将在后台运行分布训练。您可以通过output.multi_gpu.train.log文件查看结果。

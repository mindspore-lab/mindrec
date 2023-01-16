# Wide&Deep性能Benchmark
## 物理环境

CPU：Inter(R) Xeon(R) Platinum 8160@2.10GHz

GPU: Tesla V100-SXM2-16GB * 8

## 软件版本

MindSpore v1.8

CUDA Version: 10.1

Ubuntu 16.04

## 模型结构

vocab_size: 5860000

deep_embedding_vec_size: 16

hidden_units_num: 7，output dim： 1024

## 数据集

criteo(one day)


## 复现步骤


**1. 克隆代码**

```bash
git clone https://gitee.com/mindspore/recommender.git
cd benchmarks/wide_deep
```


**2. 下载数据集并进行格式转换**

详见datasets目录，使用对应脚本预处理数据，生成的MindRecord数据并存放在./data/mindrecord下。


**3. 开始训练**

- **1 GPU**

```bash
python train_and_eval.py --data_path=./data/mindrecord --device_target="GPU"
```
或
```bash
bash scripts/run_standalone_train_for_gpu.sh 5 ./data/mindrecord
```
日志中，per step time: xxx ms，表示单step平均性能为xxx ms

- **4 GPU**

```bash
bash scripts/run_distribute_train_for_gpu.sh 4 5 ./data/mindrecord 127.0.0.1 2898
```
训练日志保存在./worker_*/内，日志中，per step time: xxx ms，表示单step平均性能为xxx ms

- **8 GPU**

```bash
bash scripts/run_distribute_train_for_gpu.sh 8 5 ./data/mindrecord 127.0.0.1 2898
```
训练日志保存在./worker_*/内，日志中，per step time: xxx ms，表示单step平均性能为xxx ms


## 性能数据


| GPU Num | Throughput |
| ------- | ---------- |
| 1p      | 267,558    |
| 4p      | 767,663    |
| 8p      | 1,163,636  |

## criteo数据集下载与转换

数据集下载链接 : http://ailab.criteo.com/download-criteo-1tb-click-logs-dataset/


1. 使用 download.sh 来下载数据集并解压, 如下载并解压一天的数据，可以执行如下命令：
```bash
bash download.sh 1
```


2. 使用 process_data.py 预处理下载的原生数据集，转换为mindrecord格式

```bash
python process_data.py --data_path=~/mindspore/rec/criteo_1tb_data --part_num=1
```
表示处理一天的数据，data_path表示原始数据集存放路径，part_num表示下载下来的文件数量，即对应下载多少天的数据。


3. 如果原始数据存放在：~/mindspore/rec/criteo_1tb_data, 则转换后的mindrecord格式文件自动存放在： ~/mindspore/rec/criteo_1tb_data/mindrecord
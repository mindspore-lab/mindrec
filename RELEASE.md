# MindSpore Recommender Release Notes

## MindSpore Recommender 0.3.0 Release Notes

### Bug Fixes
- Fixed the issue that cache eviction process occasionally appeared core dump on Embedding Cache mode.
- Fixed occasional degradation of data processing performance due to resource competition in single-node 8-card GPU environment training Wide&Deep network.

### Contributors
Thanks goes to these wonderful people:
lanbingliang，lizhenyu，zhoupeichen.
Contributions of any kind are welcome!


## MindSpore Recommender 0.2.0 Release Notes

### Major Features and Improvements
- Support dynamic Embedding and real-time feature addition and deletion to realize dynamic feature admission and eviction as well as incremental recommendation models import and export.
- Support recommended model online learning and realize minute-level end-to-end incremental training + model update.
- Support HBM-DRAM-SSD multilevel distributed feature cache and realize TB-level recommendation models training.

### API Changed
- Add API `mindspore_rec.RecModel`
- Add API `mindspore_rec.HashEmbeddingLookup`

### Bug Fixes
- Fix the put operation stuck after multiple put and erase operations on MapParameter.

### Contributors
Thanks goes to these wonderful people:
hewei, chengang, lizhenyu, wangrui, zhoupeichen, gaoyong, hangangqiang, xupan, zhangxiaohan.
Contributions of any kind are welcome!

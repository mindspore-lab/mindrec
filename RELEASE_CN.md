# MindSpore Recommender Release Notes

## MindSpore Recommender 0.2.0 Release Notes

### 主要特性和增强

- 支持动态Embedding，支持特征实时增删，实现特征动态准入和淘汰以及模型增量导入导出。
- 支持推荐模型在线学习，实现分钟级端到端增量训练+模型更新。
- 支持HBM-DRAM-SSD多级分布式特征缓存，实现TB级推荐网络模型训练。

### API变更

- 新增API `mindspore_rec.RecModel`
- 新增API `mindspore_rec.HashEmbeddingLookup`

### 问题修复
- 修复MapParameter多次put和erase操作后，导致put操作卡死的问题。

### 贡献者

感谢以下开发者做出的贡献：
hewei，chengang，lizhenyu，wangrui，zhoupeichen，gaoyong，hangangqiang，xupan，zhangxiaohan。
欢迎以任何形式对项目提供贡献。

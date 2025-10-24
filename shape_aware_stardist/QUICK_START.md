# Shape-Aware StarDist - Quick Start Guide

快速开始指南，帮助您在5分钟内开始训练。

## 🎯 目标

使用Google Colab和DSB2018数据集训练Shape-aware StarDist模型。

## ⚡ 快速开始（3个步骤）

### 步骤1: 准备数据 (10-20分钟)

```bash
# 1. 配置Kaggle API
mkdir -p ~/.kaggle
mv ~/Downloads/kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json

# 2. 下载DSB2018数据集
cd /Users/zhouyitong/Downloads/hDRG-autoseg-etta_dev/shape_aware_stardist
./setup_dsb2018.sh

# 3. 准备上传到Google Drive
./upload_to_drive.sh
```

### 步骤2: 上传数据 (5-15分钟)

1. 访问 https://drive.google.com
2. 创建文件夹：`My Drive/shape_data/`
3. 上传 `data/dsb2018.zip` 到该文件夹
4. 等待上传完成

### 步骤3: 在Colab训练 (3-17小时)

1. 访问 https://colab.research.google.com
2. 从GitHub打开notebook：
   - 仓库：`EttaZhouuuuu/hdrg-adaptive-stardist`
   - 分支：`feature/shape-aware-backbone`
   - 文件：`shape_aware_stardist/colab/train_on_colab.ipynb`
3. 设置GPU：Runtime → Change runtime type → GPU
4. 运行所有单元格

## 📚 详细文档

- **完整训练指南**: [COLAB_TRAINING_GUIDE.md](COLAB_TRAINING_GUIDE.md)
- **数据集设置**: [DSB2018_SETUP.md](DSB2018_SETUP.md)
- **项目README**: [README.md](README.md)

## 🔧 系统要求

### 本地（数据准备）
- macOS / Linux / Windows
- Python 3.8+
- 2GB可用磁盘空间
- 网络连接（下载数据）

### Colab（训练）
- Google账号
- 2GB Google Drive空间
- Colab Pro（推荐，更快的GPU）

## ⏱️ 时间估计

| 步骤 | 时间 |
|------|------|
| 下载数据 | 5-15分钟 |
| 准备上传 | 2-5分钟 |
| 上传到Drive | 5-15分钟 |
| Colab训练 | 3-17小时 |

## 💡 提示

1. **Kaggle API**: 从 https://www.kaggle.com/settings 获取
2. **Colab Pro**: 推荐使用以获得更好的GPU（A100/V100）
3. **训练时间**: 使用A100 GPU约3.5小时，T4约17小时
4. **自动保存**: 训练结果自动保存到Google Drive

## ❓ 遇到问题？

### Kaggle API不工作
```bash
# 检查配置
ls -la ~/.kaggle/kaggle.json
# 应该显示: -rw------- (600权限)
```

### 数据上传失败
- 使用Google Drive桌面客户端
- 或在Colab中直接下载（需要Kaggle API）

### Colab内存不足
```python
# 在notebook中减小batch size
batch_size = 4  # 默认是8
```

### 训练中断
- 使用检查点恢复训练
- 或使用Colab Pro获得更长的运行时间

## 🎓 学习资源

- [StarDist论文](https://arxiv.org/abs/1806.03535)
- [Transformer架构](https://arxiv.org/abs/1706.03762)
- [Colab教程](https://colab.research.google.com/notebooks/intro.ipynb)

## 📞 获取帮助

- 查看 [COLAB_TRAINING_GUIDE.md](COLAB_TRAINING_GUIDE.md) 的常见问题部分
- 检查 [DSB2018_SETUP.md](DSB2018_SETUP.md) 的故障排除
- 查看GitHub Issues

## 🚀 下一步

训练完成后：
1. 下载训练好的模型
2. 在测试集上评估
3. 可视化预测结果
4. 尝试不同的超参数

祝您训练顺利！🎉

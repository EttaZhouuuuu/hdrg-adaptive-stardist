# DSB2018 Dataset Setup Guide

本指南将帮助您下载、准备和使用DSB2018数据集进行训练。

## 前置要求

1. Kaggle账号
2. Kaggle API token
3. Python 3.8+

## 步骤1: 设置Kaggle API

### 1.1 获取API Token

1. 访问 https://www.kaggle.com/settings
2. 向下滚动到"API"部分
3. 点击"Create New API Token"
4. 下载`kaggle.json`文件

### 1.2 配置API Token

```bash
# 创建.kaggle目录
mkdir -p ~/.kaggle

# 移动kaggle.json到.kaggle目录
mv ~/Downloads/kaggle.json ~/.kaggle/

# 设置正确的权限
chmod 600 ~/.kaggle/kaggle.json
```

## 步骤2: 下载和准备数据集

### 2.1 安装依赖

```bash
cd /Users/zhouyitong/Downloads/hDRG-autoseg-etta_dev/shape_aware_stardist
pip install kaggle scikit-image tqdm
```

### 2.2 下载数据集

```bash
# 下载并准备DSB2018数据集
python -m shape_aware_stardist.data.download_dsb2018 --download
```

这个命令会：
1. 从Kaggle下载DSB2018数据集（约1.5GB）
2. 解压数据
3. 处理图像和掩码
4. 创建训练/验证/测试分割
5. 生成实例分割掩码

### 2.3 验证数据集

```bash
# 测试数据加载
python -m shape_aware_stardist.data.dsb2018_dataset
```

## 步骤3: 数据集结构

下载和处理后，数据集结构如下：

```
data/dsb2018/
├── train/
│   ├── images/          # 训练图像
│   └── masks/           # 训练掩码
├── val/
│   ├── images/          # 验证图像
│   └── masks/           # 验证掩码
└── test/
    ├── images/          # 测试图像
    └── masks/           # 测试掩码
```

## 步骤4: 使用数据集训练

### 4.1 本地训练

```python
from shape_aware_stardist.data.dsb2018_dataset import get_dsb2018_loaders
from shape_aware_stardist.models import AdaptiveShapeEncoder
from shape_aware_stardist.training import ShapeAwareTrainer

# 加载数据
train_loader, val_loader, test_loader = get_dsb2018_loaders(
    batch_size=8,
    input_size=(256, 256),
    n_rays=32
)

# 创建模型
model = AdaptiveShapeEncoder(
    in_channels=3,
    n_rays=32,
    base_channels=64
)

# 训练
trainer = ShapeAwareTrainer(...)
trainer.train(num_epochs=200)
```

### 4.2 在Colab上训练

1. 上传数据到Google Drive：
```bash
# 压缩处理后的数据
cd /Users/zhouyitong/Downloads/hDRG-autoseg-etta_dev/data
zip -r dsb2018.zip dsb2018/

# 上传dsb2018.zip到Google Drive的shape_data文件夹
```

2. 在Colab中：
```python
# 挂载Drive
from google.colab import drive
drive.mount('/content/drive')

# 解压数据
!unzip /content/drive/MyDrive/shape_data/dsb2018.zip -d data/
```

## 数据集统计

### 训练集
- 图像数量: ~530
- 平均每张图像的细胞核数量: ~30
- 图像大小: 变化（最小256x256，最大1024x1024）

### 验证集
- 图像数量: ~135
- 用于模型选择和早停

### 测试集
- 图像数量: ~65
- 用于最终评估

## 常见问题

### Q1: Kaggle API认证失败
**A**: 确保kaggle.json在正确位置且权限正确：
```bash
ls -la ~/.kaggle/kaggle.json
# 应该显示: -rw------- (600权限)
```

### Q2: 下载速度慢
**A**: 可以使用代理或VPN加速Kaggle下载

### Q3: 内存不足
**A**: 减小batch_size或input_size：
```python
train_loader, val_loader, test_loader = get_dsb2018_loaders(
    batch_size=4,  # 减小batch size
    input_size=(128, 128)  # 减小图像大小
)
```

### Q4: 数据增强太强
**A**: 在配置文件中调整增强参数：
```python
# 编辑 configs/dsb2018_config.py
PROCESSING_CONFIG = {
    'augmentation': {
        'enabled': True,
        'rotation_range': (-90, 90),  # 减小旋转范围
        'scale_range': (0.9, 1.1),    # 减小缩放范围
        ...
    }
}
```

## 下一步

1. 查看训练脚本: `examples/train_dsb2018.py`
2. 查看Colab notebook: `colab/train_on_colab.ipynb`
3. 阅读完整文档: `README.md`

## 参考资料

- DSB2018竞赛: https://www.kaggle.com/c/data-science-bowl-2018
- StarDist论文: https://arxiv.org/abs/1806.03535
- 项目文档: https://github.com/YOUR_USERNAME/shape-aware-stardist

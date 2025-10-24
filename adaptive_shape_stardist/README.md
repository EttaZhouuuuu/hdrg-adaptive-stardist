# Adaptive Shape StarDist

## 项目概述

Adaptive Shape StarDist 是一个创新的细胞分割框架，旨在改进原始StarDist方法中固定射线表示的局限性。

### 核心创新点

#### 1. **自适应形状编码器 (Adaptive Shape Encoder)**
- 替代固定数量的射线，使用可学习的形状表示
- 能够动态调整采样点的数量和位置
- 更好地处理高度不规则的细胞形状

#### 2. **可变形卷积 (Deformable Convolution)**
- 学习自适应的采样位置偏移
- 根据细胞形状特征动态调整感受野
- 提高对不规则边界的建模能力

#### 3. **形状先验知识编码 (Shape Prior Encoding)**
- 集成领域特定的形状知识
- 使用attention机制融合全局和局部特征
- 提高对特定细胞类型的识别准确率

## 架构设计

```
Input Image
    ↓
[Backbone Network] ────→ Multi-scale Features
    ↓                           ↓
[Shape Prior Encoder] ←─────────┘
    ↓
[Adaptive Shape Encoder]
    ↓
[Deformable Sampling Points]
    ↓
[Shape Reconstruction]
    ↓
Instance Masks
```

## 主要特性

- ✨ **灵活的形状表示**: 不再局限于固定数量的射线
- 🎯 **自适应采样**: 根据形状复杂度动态调整采样点
- 🧠 **形状先验**: 集成领域知识提高准确性
- 📊 **多尺度融合**: 有效处理不同尺寸的细胞
- 🚀 **高效训练**: 端到端可微分架构

## 技术细节

### 可变形卷积
```python
# 学习偏移量和权重
offset = offset_conv(features)
mask = mask_conv(features)
deformed_features = deformable_conv(features, offset, mask)
```

### 自适应采样点
```python
# 基于特征图生成采样点
num_points = adaptive_point_predictor(features)  # 可变数量
positions = position_predictor(features)          # 自适应位置
```

### 形状先验编码
```python
# 使用transformer编码全局形状信息
shape_prior = shape_encoder(global_features)
enhanced_features = attention_fusion(local_features, shape_prior)
```

## 与原始StarDist的对比

| 特性 | StarDist | Adaptive Shape StarDist |
|------|----------|------------------------|
| 边界表示 | 固定N条射线 | 自适应采样点 |
| 形状灵活性 | 受限于星凸多边形 | 任意形状 |
| 复杂边界处理 | 较弱 | 强 |
| 先验知识 | 无 | 可集成 |
| 参数效率 | 中等 | 高 |

## 安装

```bash
cd adaptive_shape_stardist
pip install -r requirements.txt
pip install -e .
```

## 快速开始

### 训练模型

```python
from adaptive_shape_stardist import AdaptiveShapeStarDist
from adaptive_shape_stardist.configs import AdaptiveConfig

# 配置模型
config = AdaptiveConfig(
    n_channel_in=1,
    min_sampling_points=32,
    max_sampling_points=128,
    use_shape_prior=True,
    deformable_groups=4
)

# 创建模型
model = AdaptiveShapeStarDist(config, name='my_model', basedir='models/')

# 训练
model.train(X_train, Y_train, validation_data=(X_val, Y_val))
```

### 推理

```python
# 加载模型
model = AdaptiveShapeStarDist.from_pretrained('models/my_model')

# 预测
labels, details = model.predict_instances(image)
```

## 项目结构

```
adaptive_shape_stardist/
├── core/                      # 核心组件
│   ├── deformable_conv.py    # 可变形卷积实现
│   ├── shape_encoder.py      # 自适应形状编码器
│   ├── shape_prior.py        # 形状先验编码器
│   └── sampling.py           # 自适应采样
├── models/                    # 模型定义
│   ├── adaptive_shape_model.py
│   └── backbone.py
├── training/                  # 训练相关
│   ├── loss.py               # 损失函数
│   └── trainer.py            # 训练器
├── inference/                 # 推理相关
│   └── predictor.py
├── configs/                   # 配置文件
│   └── config.py
└── utils/                     # 工具函数
    └── visualization.py
```

## 实验结果

待补充...

## 引用

如果您使用了这个项目，请引用：

```bibtex
@software{adaptive_shape_stardist,
  title={Adaptive Shape StarDist: Flexible Cell Segmentation with Deformable Convolutions},
  author={Zhou, Yitong},
  year={2025},
  version={0.1.0}
}
```

## 许可证

MIT License

## 致谢

本项目基于原始StarDist框架进行创新和改进。


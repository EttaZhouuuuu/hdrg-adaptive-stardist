# Google Colab Training Guide

完整的Colab训练指南，包括数据准备、上传和训练步骤。

## 📋 前置要求

1. Google账号（有Colab访问权限）
2. Google Drive空间（至少2GB）
3. 已下载并准备好的DSB2018数据集

## 步骤1: 准备数据

### 1.1 本地准备数据

如果您还没有下载DSB2018数据集，请先运行：

```bash
cd /Users/zhouyitong/Downloads/hDRG-autoseg-etta_dev/shape_aware_stardist
./setup_dsb2018.sh
```

### 1.2 压缩数据

```bash
cd /Users/zhouyitong/Downloads/hDRG-autoseg-etta_dev/data
zip -r dsb2018.zip dsb2018/
```

压缩后的文件大小约为：**~500MB**

## 步骤2: 上传数据到Google Drive

### 方法1: 通过网页上传（推荐）

1. 访问 https://drive.google.com
2. 创建文件夹结构：
   ```
   My Drive/
   └── shape_data/
       └── dsb2018.zip  (上传这里)
   ```
3. 上传 `dsb2018.zip` 到 `shape_data` 文件夹

### 方法2: 使用Google Drive桌面客户端

1. 安装Google Drive桌面应用
2. 将 `dsb2018.zip` 复制到：
   ```
   ~/Google Drive/shape_data/dsb2018.zip
   ```
3. 等待同步完成

### 方法3: 使用命令行（需要rclone）

```bash
# 安装rclone
brew install rclone

# 配置Google Drive
rclone config

# 上传文件
rclone copy dsb2018.zip gdrive:shape_data/
```

## 步骤3: 推送代码到GitHub

确保您的代码已经推送到GitHub：

```bash
cd /Users/zhouyitong/Downloads/hDRG-autoseg-etta_dev
git push origin feature/shape-aware-backbone
```

## 步骤4: 在Colab中训练

### 4.1 打开Notebook

1. 访问 https://colab.research.google.com
2. 选择 **GitHub** 标签
3. 输入仓库URL：`https://github.com/EttaZhouuuuu/hdrg-adaptive-stardist`
4. 选择分支：`feature/shape-aware-backbone`
5. 打开：`shape_aware_stardist/colab/train_on_colab.ipynb`

### 4.2 设置GPU

1. 点击菜单：**Runtime** → **Change runtime type**
2. 选择：**Hardware accelerator: GPU**
3. 如果您有Colab Pro，选择：**GPU type: A100** 或 **V100**
4. 点击 **Save**

### 4.3 运行训练

按顺序运行所有单元格：

1. **检查GPU** - 确认GPU可用
2. **挂载Drive** - 授权访问Google Drive
3. **克隆仓库** - 下载代码
4. **安装依赖** - 安装必要的包
5. **复制数据** - 从Drive复制到本地（加快训练速度）
6. **导入库** - 导入所需模块
7. **加载数据** - 创建数据加载器
8. **创建模型** - 初始化模型
9. **设置训练** - 配置损失函数和优化器
10. **创建训练器** - 初始化训练器
11. **开始训练** - 运行训练循环
12. **保存结果** - 将结果保存回Drive

## 📊 训练配置

### 默认配置

```python
batch_size = 8          # 根据GPU内存调整
input_size = (256, 256) # 图像大小
n_rays = 32            # StarDist射线数
learning_rate = 1e-4   # 学习率
num_epochs = 200       # 最大训练轮数
```

### GPU内存优化

如果遇到内存不足（OOM）错误：

```python
# 减小batch size
batch_size = 4  # 或 2

# 减小图像大小
input_size = (128, 128)

# 减小模型大小
base_channels = 32  # 默认是64
```

### 加速训练

如果您有Colab Pro：

```python
# 使用混合精度训练
from torch.cuda.amp import autocast, GradScaler
scaler = GradScaler()

# 增大batch size
batch_size = 16  # A100可以支持更大的batch
```

## ⏱️ 预期训练时间

| GPU类型 | Batch Size | 每个Epoch | 总时间(200 epochs) |
|---------|-----------|-----------|-------------------|
| T4 (Free) | 8 | ~5分钟 | ~17小时 |
| V100 (Pro) | 8 | ~2分钟 | ~7小时 |
| A100 (Pro) | 16 | ~1分钟 | ~3.5小时 |

**注意**: Colab Free有使用时间限制（~12小时），建议：
- 使用Colab Pro进行长时间训练
- 或分多次训练，使用检查点恢复

## 📈 监控训练

### 在Colab中查看

训练过程中会显示：
- 当前epoch和总epoch数
- 训练损失和验证损失
- 学习率变化
- 最佳模型保存信息

### 使用TensorBoard

```python
# 在新单元格中运行
%load_ext tensorboard
%tensorboard --logdir /content/experiments/tensorboard
```

## 💾 保存和下载结果

### 自动保存到Drive

训练完成后，结果会自动保存到：
```
My Drive/shape_aware_stardist/trained_models/
├── model_best.pth          # 最佳模型权重
├── training_state.pth      # 完整训练状态
├── training.log            # 训练日志
└── tensorboard/            # TensorBoard日志
```

### 下载到本地

```python
# 压缩结果
!cd /content/drive/MyDrive/shape_aware_stardist && \
 zip -r trained_models.zip trained_models/

# 从Drive下载 trained_models.zip
```

## 🔄 恢复训练

如果训练中断，可以从检查点恢复：

```python
# 加载检查点
checkpoint = torch.load('/content/experiments/training_state.pth')
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
start_epoch = checkpoint['epoch'] + 1

# 继续训练
trainer.train(num_epochs=200, start_epoch=start_epoch)
```

## ❓ 常见问题

### Q1: 上传数据太慢怎么办？

**A**: 
- 使用Google Drive桌面客户端
- 或在Colab中直接下载（如果有Kaggle API）
- 压缩数据以减小文件大小

### Q2: Colab断开连接怎么办？

**A**:
- 使用Colab Pro获得更长的连接时间
- 定期保存检查点
- 使用浏览器插件保持连接

### Q3: GPU内存不足怎么办？

**A**:
```python
# 减小batch size
batch_size = 4

# 或减小模型
base_channels = 32
n_blocks = 3
```

### Q4: 训练太慢怎么办？

**A**:
- 升级到Colab Pro使用更好的GPU
- 减小图像大小
- 减少数据增强

### Q5: 如何验证训练效果？

**A**:
```python
# 在测试集上评估
from shape_aware_stardist.evaluation.metrics import SegmentationMetrics

metrics = SegmentationMetrics(n_rays=32)
# 运行评估...
```

## 📚 相关资源

- [Colab官方文档](https://colab.research.google.com/notebooks/intro.ipynb)
- [PyTorch教程](https://pytorch.org/tutorials/)
- [项目README](../README.md)
- [DSB2018数据集文档](../DSB2018_SETUP.md)

## 🎯 下一步

训练完成后：
1. 下载训练好的模型
2. 在测试集上评估
3. 可视化预测结果
4. 调整超参数重新训练

祝训练顺利！🚀

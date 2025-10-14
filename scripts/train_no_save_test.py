"""
不保存权重的简化训练测试
"""
import argparse
import os
import numpy as np
from skimage import io
from csbdeep.utils import normalize_mi_ma
from stardist.models import StarDist2D, Config2D

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--slides", type=str, required=True) 
    parser.add_argument("--n_epochs", type=int, default=2)
    parser.add_argument("--max_samples", type=int, default=6)
    return parser.parse_args()

def load_data(data_path, slides, max_samples=6):
    """加载数据"""
    X, Y = [], []
    
    slide = slides.strip()
    patches_dir = os.path.join(data_path, slide, 'patches_2048')
    masks_dir = os.path.join(data_path, slide, 'pseudo_gt_masks_2048')
    
    patch_files = sorted([f for f in os.listdir(patches_dir) if f.endswith('.png')])
    
    for i, patch_file in enumerate(patch_files[:max_samples]):
        img_path = os.path.join(patches_dir, patch_file)
        mask_path = os.path.join(masks_dir, patch_file)
        
        if not os.path.exists(mask_path):
            continue
            
        img = io.imread(img_path)
        mask = io.imread(mask_path)
        
        # 调整到更小的尺寸以快速测试
        from skimage.transform import resize
        img_small = resize(img, (128, 128), preserve_range=True).astype(np.float32)
        mask_small = resize(mask, (128, 128), preserve_range=True).astype(np.uint8)
        
        # 归一化
        img_small = normalize_mi_ma(img_small, 0, 255)
        mask_small = (mask_small // 255).astype(np.uint8)
        
        X.append(img_small)
        Y.append(mask_small)
        
        print(f"  加载样本 {i+1}: {img_small.shape}, 范围 [{img_small.min():.3f}, {img_small.max():.3f}]")
    
    return np.array(X), np.array(Y)

def main():
    args = parse_args()
    
    print("=== 无保存训练测试 ===")
    
    # 加载数据
    X, Y = load_data(args.data_path, args.slides, args.max_samples)
    print(f"数据形状: X {X.shape}, Y {Y.shape}")
    
    # 创建配置 - 不保存权重
    conf = Config2D(
        n_rays=16,  # 减少rays以加快训练
        grid=(1, 1),  # 简化grid
        n_channel_in=3,
        train_patch_size=(64, 64),  # 更小的patch
        train_batch_size=2,
        train_learning_rate=0.001,
        train_epochs=args.n_epochs,
        train_steps_per_epoch=2,  # 只训练2步
        train_save_every_epochs=1000  # 设置很大的数字避免保存
    )
    
    print(f"配置: n_rays={conf.n_rays}, patch_size={conf.train_patch_size}")
    
    # 创建临时目录
    temp_dir = "/tmp/stardist_test"
    os.makedirs(temp_dir, exist_ok=True)
    
    # 创建模型
    model = StarDist2D(conf, name='no_save_test', basedir=temp_dir)
    
    print("开始训练...")
    try:
        # 使用内置训练方法，但避免保存
        history = model.train(X, Y, validation_data=(X[:2], Y[:2]))
        print("✅ 训练成功完成！")
        print(f"训练历史: {list(history.history.keys())}")
        
        # 手动测试一下预测
        print("测试预测...")
        labels, _ = model.predict_instances(X[0])
        print(f"预测标签形状: {labels.shape}, 唯一值: {np.unique(labels)}")
        
    except Exception as e:
        print(f"❌ 训练失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

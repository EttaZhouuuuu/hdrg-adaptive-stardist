"""
简化的测试训练脚本 - 验证基本功能
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
    parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument("--n_epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--max_samples", type=int, default=20)
    return parser.parse_args()

def load_data(data_path, slides, max_samples=20):
    """加载数据 - 简化版本"""
    X, Y = [], []
    
    slide = slides.strip()
    patches_dir = os.path.join(data_path, slide, 'patches_2048')
    masks_dir = os.path.join(data_path, slide, 'pseudo_gt_masks_2048')
    
    patch_files = sorted([f for f in os.listdir(patches_dir) if f.endswith('.png')])
    
    for i, patch_file in enumerate(patch_files[:max_samples]):
        if i >= max_samples:
            break
            
        img_path = os.path.join(patches_dir, patch_file)
        mask_path = os.path.join(masks_dir, patch_file)
        
        if not os.path.exists(mask_path):
            continue
            
        img = io.imread(img_path)
        mask = io.imread(mask_path)
        
        # 调整大小到可管理的尺寸
        from skimage.transform import resize
        img_small = resize(img, (256, 256), preserve_range=True).astype(np.float32)
        mask_small = resize(mask, (256, 256), preserve_range=True).astype(np.uint8)
        
        # 归一化
        img_small = normalize_mi_ma(img_small, 0, 255)
        mask_small = (mask_small // 255).astype(np.uint8)
        
        X.append(img_small)
        Y.append(mask_small)
        
        if (i + 1) % 5 == 0:
            print(f"  已加载 {i + 1} 个样本...")
    
    return np.array(X), np.array(Y)

def main():
    args = parse_args()
    
    print("=== 简化训练测试 ===")
    print(f"最大样本数: {args.max_samples}")
    
    # 加载数据
    X, Y = load_data(args.data_path, args.slides, args.max_samples)
    print(f"加载完成: X {X.shape}, Y {Y.shape}")
    
    # 创建基础StarDist配置
    conf = Config2D(
        n_rays=32,
        grid=(2, 2),
        n_channel_in=3 if len(X.shape) == 4 and X.shape[-1] == 3 else 1,
        train_patch_size=(128, 128),
        train_batch_size=args.batch_size,
        train_learning_rate=0.001,
        train_epochs=args.n_epochs,
        train_steps_per_epoch=min(10, len(X) // args.batch_size)
    )
    
    print(f"配置: n_rays={conf.n_rays}, patch_size={conf.train_patch_size}")
    
    # 创建模型
    model = StarDist2D(conf, name='test_stardist', basedir=args.ckpt_path)
    
    # 训练
    print("开始训练...")
    model.train(X, Y, validation_data=(X[:2], Y[:2]))
    
    print("训练完成！")

if __name__ == "__main__":
    main()

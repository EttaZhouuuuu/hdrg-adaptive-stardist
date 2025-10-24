"""
测试数据加载和处理的脚本
"""
import os
import numpy as np
from skimage import io
from csbdeep.utils import normalize_mi_ma

def load_and_check_data():
    """测试数据加载，检查形状和类型"""
    # 使用本地路径进行测试
    data_path = "./data"  # 你可以修改为实际的数据路径
    slide = "240819_Ji_N1_H_EScan"
    
    patches_dir = os.path.join(data_path, slide, 'patches_2048')
    masks_dir = os.path.join(data_path, slide, 'pseudo_gt_masks_2048')
    
    print(f"检查目录:")
    print(f"Patches: {patches_dir}")
    print(f"Masks: {masks_dir}")
    
    # 检查目录是否存在
    if not os.path.exists(patches_dir):
        print(f"Patches目录不存在: {patches_dir}")
        return
    if not os.path.exists(masks_dir):
        print(f"Masks目录不存在: {masks_dir}")
        return
    
    # 获取文件列表
    patch_files = sorted([f for f in os.listdir(patches_dir) if f.endswith('.png')])
    print(f"找到 {len(patch_files)} 个patch文件")
    
    # 检查前几个文件的形状
    X, Y = [], []
    for i, patch_file in enumerate(patch_files[:5]):  # 只检查前5个文件
        img_path = os.path.join(patches_dir, patch_file)
        mask_path = os.path.join(masks_dir, patch_file)
        
        if not os.path.exists(mask_path):
            print(f"缺少mask: {patch_file}")
            continue
            
        img = io.imread(img_path)
        mask = io.imread(mask_path)
        
        print(f"\n文件 {i+1}: {patch_file}")
        print(f"  原始图像形状: {img.shape}, 类型: {img.dtype}")
        print(f"  原始mask形状: {mask.shape}, 类型: {mask.dtype}")
        
        # 标准化
        img_norm = normalize_mi_ma(img, 0, 255)
        mask_binary = (mask // 255).astype(np.uint8)
        
        print(f"  处理后图像形状: {img_norm.shape}, 类型: {img_norm.dtype}")
        print(f"  处理后mask形状: {mask_binary.shape}, 类型: {mask_binary.dtype}")
        print(f"  图像值范围: [{img_norm.min():.3f}, {img_norm.max():.3f}]")
        print(f"  Mask唯一值: {np.unique(mask_binary)}")
        
        X.append(img_norm)
        Y.append(mask_binary)
    
    if X and Y:
        print(f"\n最终检查:")
        print(f"所有图像形状一致: {all(x.shape == X[0].shape for x in X)}")
        print(f"所有mask形状一致: {all(y.shape == Y[0].shape for y in Y)}")
        
        # 尝试转换为numpy数组
        try:
            X_array = np.array(X)
            Y_array = np.array(Y)
            print(f"成功转换为数组:")
            print(f"X数组形状: {X_array.shape}")
            print(f"Y数组形状: {Y_array.shape}")
        except Exception as e:
            print(f"转换为数组失败: {e}")
            return X, Y
    
    return X, Y

if __name__ == "__main__":
    X, Y = load_and_check_data()

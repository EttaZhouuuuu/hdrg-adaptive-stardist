"""
Training script for the Adaptive StarDist model - Test Version
简化的测试版本，用于调试数据流程
"""

import argparse
import os
import numpy as np
from skimage import io
from csbdeep.utils import normalize_mi_ma

def parse_args():
    parser = argparse.ArgumentParser(description="Train Multi-Scale Adaptive StarDist model - Test")
    parser.add_argument("--data_path", type=str, default="./data",
                       help="Path to training data directory")
    parser.add_argument("--slides", type=str, default="240819_Ji_N1_H_EScan",
                       help="Comma-separated list of slide names")
    parser.add_argument("--n_epochs", type=int, default=1,
                       help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=2,
                       help="Training batch size")
    return parser.parse_args()

def load_training_data(data_path, slides, max_samples=10):
    """Load and prepare training data - 限制样本数量用于测试"""
    X, Y = [], []
    
    print(f"Loading data from: {data_path}")
    print(f"Slides to process: {slides}")
    
    if isinstance(slides, str):
        slides_list = slides.split(',')
    else:
        slides_list = slides
    
    # 检查数据目录
    if not os.path.exists(data_path):
        print(f"数据路径不存在: {data_path}")
        # 创建模拟数据用于测试
        print("创建模拟数据用于测试...")
        X = [np.random.rand(256, 256).astype(np.float32) for _ in range(max_samples)]
        Y = [np.random.randint(0, 2, (256, 256)).astype(np.uint8) for _ in range(max_samples)]
        return np.array(X), np.array(Y)
    
    available_slides = [name for name in os.listdir(data_path) 
                       if os.path.isdir(os.path.join(data_path, name))]
    print(f"Available slides: {available_slides}")
    
    sample_count = 0
    for slide in slides_list:
        slide = slide.strip()
        if slide not in available_slides:
            print(f"Warning: Slide '{slide}' not found in {data_path}")
            continue
            
        patches_dir = os.path.join(data_path, slide, 'patches_2048')
        masks_dir = os.path.join(data_path, slide, 'pseudo_gt_masks_2048')
        
        print(f"Processing slide {slide}:")
        print(f"  Patches dir: {patches_dir}")
        print(f"  Masks dir: {masks_dir}")
        
        if not os.path.exists(patches_dir):
            print(f"Warning: Patches directory not found: {patches_dir}")
            continue
        if not os.path.exists(masks_dir):
            print(f"Warning: Masks directory not found: {masks_dir}")
            continue
        
        patch_files = sorted([f for f in os.listdir(patches_dir) if f.endswith('.png')])
        print(f"  Found {len(patch_files)} patch files")
        
        for patch_file in patch_files[:max_samples]:  # 限制样本数量
            img_path = os.path.join(patches_dir, patch_file)
            mask_path = os.path.join(masks_dir, patch_file)
            
            if not os.path.exists(mask_path):
                print(f"  Warning: Missing mask for {patch_file}")
                continue
                
            try:
                img = io.imread(img_path)
                mask = io.imread(mask_path)
                
                print(f"  {patch_file}: Image {img.shape} {img.dtype}, Mask {mask.shape} {mask.dtype}")
                
                # 标准化处理
                img = normalize_mi_ma(img, 0, 255).astype(np.float32)
                mask = (mask // 255).astype(np.uint8)
                
                print(f"    处理后: Image {img.shape} {img.dtype}, Mask {mask.shape} {mask.dtype}")
                
                X.append(img)
                Y.append(mask)
                sample_count += 1
                
                if sample_count >= max_samples:
                    break
                    
            except Exception as e:
                print(f"  Error loading {patch_file}: {e}")
                continue
        
        if sample_count >= max_samples:
            break
    
    if not X:
        print("没有加载到数据，创建模拟数据...")
        X = [np.random.rand(256, 256).astype(np.float32) for _ in range(max_samples)]
        Y = [np.random.randint(0, 2, (256, 256)).astype(np.uint8) for _ in range(max_samples)]
    
    X = np.array(X)
    Y = np.array(Y)
    print(f"\nFinal data shapes:")
    print(f"X: {X.shape} {X.dtype}")
    print(f"Y: {Y.shape} {Y.dtype}")
    
    return X, Y

def test_data_generator():
    """测试数据生成器"""
    print("=== 测试数据生成器 ===")
    
    # 创建模拟数据
    X = np.random.rand(5, 128, 128).astype(np.float32)
    Y = np.random.randint(0, 2, (5, 128, 128)).astype(np.uint8)
    
    print(f"模拟数据: X {X.shape}, Y {Y.shape}")
    
    # 模拟StarDistData2D的行为
    class MockStarDistData2D:
        def __init__(self, X, Y, batch_size):
            self.X = X
            self.Y = Y
            self.batch_size = batch_size
            
        def __getitem__(self, i):
            start_idx = i * self.batch_size
            end_idx = min(start_idx + self.batch_size, len(self.X))
            
            batch_x = self.X[start_idx:end_idx]
            batch_y = self.Y[start_idx:end_idx]
            
            # 模拟可能返回元组的情况
            return batch_x, batch_y
    
    mock_gen = MockStarDistData2D(X, Y, batch_size=2)
    
    # 测试数据获取
    for i in range(2):
        try:
            batch_x, batch_y = mock_gen[i]
            print(f"Batch {i}: X {batch_x.shape}, Y {batch_y.shape}")
            
            # 测试我们的处理逻辑
            def safe_convert_to_array(data):
                if isinstance(data, tuple):
                    shapes = [item.shape for item in data]
                    if len(set(shapes)) == 1:
                        return np.stack(data)
                    else:
                        return np.expand_dims(data[0], axis=0)
                return data
            
            def ensure_4d(array):
                if len(array.shape) == 3:
                    return array[..., np.newaxis]
                return array
            
            processed_x = safe_convert_to_array(batch_x)
            processed_x = ensure_4d(processed_x)
            
            print(f"  处理后: X {processed_x.shape}")
            
        except Exception as e:
            print(f"Batch {i} 失败: {e}")

def main():
    args = parse_args()
    
    print("=== 开始测试 ===")
    print(f"参数: epochs={args.n_epochs}, batch_size={args.batch_size}")
    
    # 测试数据加载
    print("\n1. 测试数据加载")
    X, Y = load_training_data(args.data_path, args.slides, max_samples=6)
    
    if len(X) == 0:
        print("没有数据可用于测试")
        return
    
    # 测试数据生成器
    print("\n2. 测试数据生成器")
    test_data_generator()
    
    print("\n=== 测试完成 ===")
    print("如果没有报错，说明数据处理逻辑是正确的")

if __name__ == "__main__":
    main()

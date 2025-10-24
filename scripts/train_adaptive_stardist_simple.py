"""
Training script for the Adaptive StarDist model - Simple Test Version
简化的测试版本，只使用numpy，用于调试数据流程
"""

import argparse
import os
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser(description="Train Multi-Scale Adaptive StarDist model - Simple Test")
    parser.add_argument("--data_path", type=str, default="./data",
                       help="Path to training data directory")
    parser.add_argument("--slides", type=str, default="240819_Ji_N1_H_EScan",
                       help="Comma-separated list of slide names")
    parser.add_argument("--n_epochs", type=int, default=1,
                       help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=2,
                       help="Training batch size")
    return parser.parse_args()

def simple_normalize(img):
    """简单的归一化函数"""
    img = img.astype(np.float32)
    img_min, img_max = img.min(), img.max()
    if img_max > img_min:
        img = (img - img_min) / (img_max - img_min)
    return img

def create_mock_data(n_samples=6, img_shape=(128, 128)):
    """创建模拟数据用于测试"""
    print(f"创建 {n_samples} 个模拟样本，形状: {img_shape}")
    
    X = []
    Y = []
    
    for i in range(n_samples):
        # 创建模拟图像（随机噪声 + 一些结构）
        img = np.random.rand(*img_shape).astype(np.float32)
        
        # 添加一些圆形结构模拟细胞
        center_y, center_x = np.random.randint(20, img_shape[0]-20), np.random.randint(20, img_shape[1]-20)
        radius = np.random.randint(5, 15)
        y, x = np.ogrid[:img_shape[0], :img_shape[1]]
        mask_circle = (x - center_x)**2 + (y - center_y)**2 <= radius**2
        img[mask_circle] += 0.5
        
        # 归一化
        img = simple_normalize(img)
        
        # 创建对应的mask
        mask = mask_circle.astype(np.uint8)
        
        X.append(img)
        Y.append(mask)
        
        print(f"  样本 {i}: 图像 {img.shape} {img.dtype} [{img.min():.3f}, {img.max():.3f}], "
              f"mask {mask.shape} {mask.dtype} {np.unique(mask)}")
    
    return np.array(X), np.array(Y)

class MockStarDistData2D:
    """模拟StarDistData2D的行为"""
    
    def __init__(self, X, Y, batch_size):
        self.X = X
        self.Y = Y
        self.batch_size = batch_size
        self.n_samples = len(X)
        print(f"MockStarDistData2D: {self.n_samples} 样本, batch_size={batch_size}")
        
    def __len__(self):
        return (self.n_samples + self.batch_size - 1) // self.batch_size
    
    def __getitem__(self, i):
        start_idx = i * self.batch_size
        end_idx = min(start_idx + self.batch_size, self.n_samples)
        
        batch_x = self.X[start_idx:end_idx]
        batch_y = self.Y[start_idx:end_idx]
        
        print(f"  Batch {i}: indices {start_idx}:{end_idx}, "
              f"X {batch_x.shape}, Y {batch_y.shape}")
        
        # 有时返回元组（模拟StarDist可能的行为）
        if np.random.random() > 0.5:
            print(f"    返回元组形式")
            return tuple(batch_x), tuple(batch_y)
        else:
            print(f"    返回数组形式")
            return batch_x, batch_y

class MultiScaleDataProcessor:
    """处理多尺度数据的类"""
    
    def __init__(self, scale_factors=[1.0, 0.5]):
        self.scale_factors = scale_factors
        print(f"MultiScaleDataProcessor: scales={scale_factors}")
    
    def safe_convert_to_array(self, data):
        """安全地将数据转换为numpy数组"""
        print(f"    转换数据: type={type(data)}")
        
        if isinstance(data, tuple):
            # 检查所有元素的形状是否一致
            shapes = [item.shape for item in data]
            print(f"      元组形状: {shapes}")
            
            if len(set(shapes)) == 1:
                # 形状一致，可以stack
                result = np.stack(data)
                print(f"      stack成功: {result.shape}")
                return result
            else:
                print(f"      形状不一致，使用第一个元素")
                return np.expand_dims(data[0], axis=0)
        elif isinstance(data, np.ndarray):
            print(f"      已是数组: {data.shape}")
            return data
        else:
            try:
                result = np.array(data)
                print(f"      转换成功: {result.shape}")
                return result
            except Exception as e:
                print(f"      转换失败: {e}")
                return data
    
    def ensure_4d(self, array):
        """确保数组是4D [batch, height, width, channels]"""
        original_shape = array.shape
        print(f"    确保4D: 原始形状 {original_shape}")
        
        if len(original_shape) == 2:
            # [H, W] -> [1, H, W, 1]
            result = array[np.newaxis, ..., np.newaxis]
        elif len(original_shape) == 3:
            # [B, H, W] -> [B, H, W, 1]
            result = array[..., np.newaxis]
        elif len(original_shape) == 4:
            result = array
        else:
            raise ValueError(f"不支持的形状: {original_shape}")
        
        print(f"      4D后形状: {result.shape}")
        return result
    
    def resize_image_simple(self, image, scale):
        """简单的图像缩放（使用最近邻插值）"""
        if scale == 1.0:
            return image
        
        batch_size, h, w, c = image.shape
        new_h, new_w = int(h * scale), int(w * scale)
        
        print(f"      缩放 {scale}: {image.shape} -> ({batch_size}, {new_h}, {new_w}, {c})")
        
        # 使用简单的最近邻插值
        resized_batch = np.zeros((batch_size, new_h, new_w, c), dtype=image.dtype)
        
        for b in range(batch_size):
            for ch in range(c):
                # 计算索引映射
                y_indices = np.clip((np.arange(new_h) / scale).astype(int), 0, h-1)
                x_indices = np.clip((np.arange(new_w) / scale).astype(int), 0, w-1)
                
                # 使用高级索引进行采样
                resized_batch[b, :, :, ch] = image[b, y_indices[:, None], x_indices, ch]
        
        return resized_batch
    
    def process_batch(self, batch):
        """处理一个批次的数据"""
        batch_x, batch_y = batch[0], batch[1]
        
        print(f"  处理批次:")
        print(f"    原始 X type: {type(batch_x)}")
        print(f"    原始 Y type: {type(batch_y)}")
        
        # 安全转换
        batch_x = self.safe_convert_to_array(batch_x)
        batch_y = self.safe_convert_to_array(batch_y)
        
        # 确保4D
        batch_x = self.ensure_4d(batch_x)
        
        # 处理多尺度
        multi_scale_x = []
        for scale in self.scale_factors:
            scaled_x = self.resize_image_simple(batch_x, scale)
            multi_scale_x.append(scaled_x)
            print(f"    尺度 {scale}: {scaled_x.shape}")
        
        return multi_scale_x, batch_y

def test_complete_pipeline():
    """测试完整的数据处理流程"""
    print("=== 测试完整数据处理流程 ===")
    
    # 1. 创建数据
    X, Y = create_mock_data(n_samples=5, img_shape=(64, 64))
    
    # 2. 创建数据生成器
    batch_size = 2
    data_gen = MockStarDistData2D(X, Y, batch_size)
    
    # 3. 创建多尺度处理器
    processor = MultiScaleDataProcessor(scale_factors=[1.0, 0.5, 0.25])
    
    # 4. 测试数据生成
    print(f"\n数据生成器有 {len(data_gen)} 个批次")
    
    for i in range(len(data_gen)):
        print(f"\n--- Batch {i} ---")
        try:
            batch = data_gen[i]
            multi_scale_x, batch_y = processor.process_batch(batch)
            
            print(f"成功处理批次 {i}:")
            print(f"  返回 {len(multi_scale_x)} 个尺度的数据")
            print(f"  Y形状: {batch_y.shape}")
            
        except Exception as e:
            print(f"批次 {i} 处理失败: {e}")
            import traceback
            traceback.print_exc()
            break
    
    print("\n=== 流程测试完成 ===")

def main():
    args = parse_args()
    
    print("=== 简化数据处理测试 ===")
    print(f"参数: epochs={args.n_epochs}, batch_size={args.batch_size}")
    
    test_complete_pipeline()
    
    print("\n如果没有错误，说明数据处理逻辑可以工作！")
    print("接下来可以将这个逻辑集成到实际的训练代码中。")

if __name__ == "__main__":
    main()

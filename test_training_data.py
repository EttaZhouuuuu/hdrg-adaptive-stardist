"""
测试训练数据生成器的脚本
"""
import numpy as np
import tensorflow as tf
from typing import List, Tuple

class SimpleStarDistData2D:
    """简化的数据生成器用于测试"""
    
    def __init__(self, X, Y, batch_size=2):
        self.X = X
        self.Y = Y
        self.batch_size = batch_size
        self.n_samples = len(X)
        
    def __len__(self):
        return self.n_samples // self.batch_size
        
    def __getitem__(self, i):
        """获取一个批次的数据"""
        start_idx = i * self.batch_size
        end_idx = min(start_idx + self.batch_size, self.n_samples)
        
        batch_x = self.X[start_idx:end_idx]
        batch_y = self.Y[start_idx:end_idx]
        
        print(f"Batch {i}:")
        print(f"  batch_x type: {type(batch_x)}, shape: {batch_x.shape}")
        print(f"  batch_y type: {type(batch_y)}, shape: {batch_y.shape}")
        
        return batch_x, batch_y

class MultiScaleStarDistData2D:
    """多尺度数据生成器测试版本"""
    
    def __init__(self, X, Y, batch_size=2, scale_factors=[1.0, 0.5]):
        self.base_generator = SimpleStarDistData2D(X, Y, batch_size)
        self.scale_factors = scale_factors
        
    def __len__(self):
        return len(self.base_generator)
        
    def __getitem__(self, i):
        """获取多尺度数据"""
        batch_x, batch_y = self.base_generator[i]
        
        print(f"\n处理多尺度数据 - Batch {i}:")
        print(f"输入数据形状: {batch_x.shape}")
        
        # 确保是4D
        if len(batch_x.shape) == 3:
            batch_x = batch_x[..., np.newaxis]
            print(f"添加channel维度后: {batch_x.shape}")
        
        multi_scale_x = []
        for scale in self.scale_factors:
            if scale == 1.0:
                scaled_x = batch_x
            else:
                # 使用numpy进行resize（避免TensorFlow图模式问题）
                scaled_x = self._resize_numpy(batch_x, scale)
            
            print(f"  尺度 {scale}: {scaled_x.shape}")
            multi_scale_x.append(scaled_x)
        
        return multi_scale_x, batch_y
    
    def _resize_numpy(self, batch_x, scale):
        """使用numpy进行图像缩放"""
        from skimage.transform import resize
        
        batch_size, h, w, c = batch_x.shape
        new_h = int(h * scale)
        new_w = int(w * scale)
        
        scaled_batch = np.zeros((batch_size, new_h, new_w, c), dtype=batch_x.dtype)
        
        for i in range(batch_size):
            for j in range(c):
                scaled_batch[i, :, :, j] = resize(
                    batch_x[i, :, :, j], 
                    (new_h, new_w),
                    preserve_range=True,
                    anti_aliasing=True
                )
        
        return scaled_batch

def test_data_generator():
    """测试数据生成器"""
    # 创建模拟数据
    batch_size = 2
    n_samples = 5
    img_shape = (128, 128)
    
    print("创建模拟数据...")
    X = np.random.rand(n_samples, *img_shape).astype(np.float32)
    Y = np.random.randint(0, 2, (n_samples, *img_shape)).astype(np.uint8)
    
    print(f"模拟数据形状:")
    print(f"X: {X.shape}")
    print(f"Y: {Y.shape}")
    
    # 测试基础生成器
    print("\n=== 测试基础生成器 ===")
    base_gen = SimpleStarDistData2D(X, Y, batch_size)
    
    for i in range(len(base_gen)):
        batch_x, batch_y = base_gen[i]
        print(f"Batch {i} 成功获取")
    
    # 测试多尺度生成器
    print("\n=== 测试多尺度生成器 ===")
    multi_gen = MultiScaleStarDistData2D(X, Y, batch_size, [1.0, 0.5])
    
    for i in range(len(multi_gen)):
        try:
            multi_x, batch_y = multi_gen[i]
            print(f"Batch {i} 多尺度数据获取成功")
            print(f"  返回了 {len(multi_x)} 个尺度的数据")
        except Exception as e:
            print(f"Batch {i} 失败: {e}")
            break

if __name__ == "__main__":
    test_data_generator()

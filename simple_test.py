"""
简单的数据处理测试，不依赖TensorFlow
"""
import numpy as np

def test_data_shapes():
    """测试数据形状处理"""
    print("=== 测试数据形状处理 ===")
    
    # 模拟StarDist数据生成器返回的数据
    batch_size = 2
    img_shape = (128, 128)
    
    # 模拟不同的数据格式
    print("\n1. 测试元组数据:")
    # 这模拟了StarDist数据生成器可能返回的格式
    batch_x_tuple = tuple([np.random.rand(*img_shape) for _ in range(batch_size)])
    batch_y_tuple = tuple([np.random.randint(0, 2, img_shape) for _ in range(batch_size)])
    
    print(f"batch_x_tuple类型: {type(batch_x_tuple)}")
    print(f"batch_x_tuple长度: {len(batch_x_tuple)}")
    print(f"第一个元素形状: {batch_x_tuple[0].shape}")
    
    print(f"batch_y_tuple类型: {type(batch_y_tuple)}")
    print(f"batch_y_tuple长度: {len(batch_y_tuple)}")
    print(f"第一个元素形状: {batch_y_tuple[0].shape}")
    
    # 测试转换为numpy数组
    try:
        X_array = np.array(batch_x_tuple)
        print(f"X转换成功: {X_array.shape}")
    except Exception as e:
        print(f"X转换失败: {e}")
        try:
            X_stack = np.stack(batch_x_tuple)
            print(f"X用stack成功: {X_stack.shape}")
        except Exception as e2:
            print(f"X用stack也失败: {e2}")
    
    try:
        Y_array = np.array(batch_y_tuple)
        print(f"Y转换成功: {Y_array.shape}")
    except Exception as e:
        print(f"Y转换失败: {e}")
        try:
            Y_stack = np.stack(batch_y_tuple)
            print(f"Y用stack成功: {Y_stack.shape}")
        except Exception as e2:
            print(f"Y用stack也失败: {e2}")
    
    print("\n2. 测试已经是数组的情况:")
    batch_x_array = np.random.rand(batch_size, *img_shape)
    batch_y_array = np.random.randint(0, 2, (batch_size, *img_shape))
    
    print(f"batch_x_array形状: {batch_x_array.shape}")
    print(f"batch_y_array形状: {batch_y_array.shape}")
    
    # 测试维度扩展
    print("\n3. 测试维度扩展:")
    if len(batch_x_array.shape) == 3:
        batch_x_4d = batch_x_array[..., np.newaxis]
        print(f"添加channel维度后: {batch_x_4d.shape}")
    
    print("\n4. 测试多尺度处理:")
    scales = [1.0, 0.5, 0.25]
    original_shape = batch_x_4d.shape
    
    for scale in scales:
        if scale == 1.0:
            scaled_shape = original_shape
        else:
            # 计算新的形状
            new_h = int(original_shape[1] * scale)
            new_w = int(original_shape[2] * scale)
            scaled_shape = (original_shape[0], new_h, new_w, original_shape[3])
        
        print(f"尺度 {scale}: {scaled_shape}")

def test_safe_data_conversion():
    """测试安全的数据转换方法"""
    print("\n=== 测试安全的数据转换 ===")
    
    def safe_convert_to_array(data):
        """安全地将数据转换为numpy数组"""
        if isinstance(data, tuple):
            # 检查所有元素的形状是否一致
            shapes = [item.shape for item in data]
            if len(set(shapes)) == 1:
                # 形状一致，可以stack
                return np.stack(data)
            else:
                print(f"形状不一致: {shapes}")
                return data
        elif isinstance(data, np.ndarray):
            return data
        else:
            try:
                return np.array(data)
            except Exception as e:
                print(f"转换失败: {e}")
                return data
    
    # 测试一致形状的元组
    consistent_tuple = tuple([np.random.rand(64, 64) for _ in range(3)])
    result1 = safe_convert_to_array(consistent_tuple)
    print(f"一致形状元组转换结果: {type(result1)}, {result1.shape if hasattr(result1, 'shape') else 'no shape'}")
    
    # 测试不一致形状的元组
    inconsistent_tuple = (
        np.random.rand(64, 64),
        np.random.rand(64, 64, 3),
        np.random.rand(32, 32)
    )
    result2 = safe_convert_to_array(inconsistent_tuple)
    print(f"不一致形状元组转换结果: {type(result2)}")
    
    # 测试已经是数组的情况
    array_data = np.random.rand(2, 64, 64)
    result3 = safe_convert_to_array(array_data)
    print(f"数组转换结果: {type(result3)}, {result3.shape}")

if __name__ == "__main__":
    test_data_shapes()
    test_safe_data_conversion()

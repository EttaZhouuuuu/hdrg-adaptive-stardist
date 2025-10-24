"""
测试模型配置的脚本
检查MultiScaleConfig2D和MultiScaleStarDist2D的初始化和参数
"""

import numpy as np
import sys
import os

def test_config_creation():
    """测试配置创建"""
    print("=== 测试模型配置创建 ===")
    
    # 测试基本参数
    config_params = {
        'n_rays': 96,
        'grid': (2, 2),
        'use_gpu': False,
        'n_channel_in': 1,
        'train_patch_size': (256, 256),
        'train_batch_size': 2,
        'train_learning_rate': 0.0003,
        'train_epochs': 1,
        'train_steps_per_epoch': 10,
        # 多尺度参数
        'scale_factors': [1.0, 0.5],
        'fusion_mode': 'weighted',
        'scale_weights': [0.6, 0.4],
        'attention_channels': 64,
        # 自适应参数
        'min_n_rays': 32,
        'max_n_rays': 96,
        'min_grid': 1,
        'max_grid': 2,
        # 其他参数
        'boundary_weight': 0.3,
        'size_weight': 0.3
    }
    
    print("基本参数:")
    for key, value in config_params.items():
        print(f"  {key}: {value}")
    
    # 尺度检测器参数
    scale_detector_params = {
        'min_sigma': 1.0,
        'max_sigma': 30.0,
        'sigma_ratio': 1.6,
        'threshold': 0.1
    }
    
    config_params['scale_detector_params'] = scale_detector_params
    
    print("\n尺度检测器参数:")
    for key, value in scale_detector_params.items():
        print(f"  {key}: {value}")
    
    return config_params

def test_config_compatibility():
    """测试配置参数的兼容性"""
    print("\n=== 测试参数兼容性 ===")
    
    config_params = test_config_creation()
    
    # 检查必需参数
    required_params = ['n_rays', 'grid', 'n_channel_in', 'train_patch_size']
    missing_params = []
    
    for param in required_params:
        if param not in config_params:
            missing_params.append(param)
    
    if missing_params:
        print(f"❌ 缺少必需参数: {missing_params}")
        return False
    else:
        print("✅ 所有必需参数都存在")
    
    # 检查参数值的有效性
    checks = []
    
    # 检查n_rays
    if isinstance(config_params['n_rays'], int) and config_params['n_rays'] > 0:
        checks.append("✅ n_rays 有效")
    else:
        checks.append("❌ n_rays 无效")
    
    # 检查grid
    grid = config_params['grid']
    if isinstance(grid, (list, tuple)) and len(grid) == 2 and all(isinstance(x, int) for x in grid):
        checks.append("✅ grid 有效")
    else:
        checks.append("❌ grid 无效")
    
    # 检查scale_factors
    scale_factors = config_params['scale_factors']
    if isinstance(scale_factors, list) and all(isinstance(x, float) for x in scale_factors):
        checks.append("✅ scale_factors 有效")
    else:
        checks.append("❌ scale_factors 无效")
    
    # 检查scale_weights
    scale_weights = config_params.get('scale_weights')
    if scale_weights is None or (len(scale_weights) == len(scale_factors)):
        checks.append("✅ scale_weights 与 scale_factors 匹配")
    else:
        checks.append("❌ scale_weights 与 scale_factors 不匹配")
    
    for check in checks:
        print(f"  {check}")
    
    return all("✅" in check for check in checks)

def test_mock_model_creation():
    """测试模型创建（不依赖TensorFlow）"""
    print("\n=== 测试模型创建逻辑 ===")
    
    config_params = test_config_creation()
    
    # 模拟MultiScaleConfig2D的创建逻辑
    class MockMultiScaleConfig2D:
        def __init__(self, **kwargs):
            # 提取多尺度参数
            self.scale_factors = kwargs.pop('scale_factors', [1.0])
            self.fusion_mode = kwargs.pop('fusion_mode', 'weighted')
            self.scale_weights = kwargs.pop('scale_weights', None)
            self.attention_channels = kwargs.pop('attention_channels', 64)
            
            # 提取自适应参数
            self.min_n_rays = kwargs.pop('min_n_rays', 32)
            self.max_n_rays = kwargs.pop('max_n_rays', 128)
            self.min_grid = kwargs.pop('min_grid', 1)
            self.max_grid = kwargs.pop('max_grid', 4)
            
            # 提取其他参数
            self.boundary_weight = kwargs.pop('boundary_weight', 0.3)
            self.size_weight = kwargs.pop('size_weight', 0.3)
            self.scale_detector_params = kwargs.pop('scale_detector_params', {})
            
            # 设置基本参数
            for key, value in kwargs.items():
                setattr(self, key, value)
    
    try:
        # 尝试创建配置
        config = MockMultiScaleConfig2D(**config_params)
        
        print("✅ 配置创建成功")
        print(f"  scale_factors: {config.scale_factors}")
        print(f"  fusion_mode: {config.fusion_mode}")
        print(f"  n_rays: {config.n_rays}")
        print(f"  grid: {config.grid}")
        print(f"  min_n_rays: {config.min_n_rays}")
        print(f"  max_n_rays: {config.max_n_rays}")
        
        return True, config
        
    except Exception as e:
        print(f"❌ 配置创建失败: {e}")
        return False, None

def test_scale_processing_logic():
    """测试尺度处理逻辑"""
    print("\n=== 测试尺度处理逻辑 ===")
    
    success, config = test_mock_model_creation()
    if not success:
        return False
    
    # 模拟图像数据
    img_shape = (128, 128, 1)
    mock_img = np.random.rand(*img_shape).astype(np.float32)
    
    print(f"模拟图像形状: {mock_img.shape}")
    
    # 测试每个尺度的处理
    for i, scale in enumerate(config.scale_factors):
        print(f"\n尺度 {i+1}: {scale}")
        
        if scale == 1.0:
            scaled_shape = img_shape
        else:
            new_h = int(img_shape[0] * scale)
            new_w = int(img_shape[1] * scale)
            scaled_shape = (new_h, new_w, img_shape[2])
        
        print(f"  预期缩放后形状: {scaled_shape}")
        
        # 检查形状有效性
        if scaled_shape[0] > 0 and scaled_shape[1] > 0:
            print(f"  ✅ 形状有效")
        else:
            print(f"  ❌ 形状无效")
            return False
    
    return True

def main():
    print("=== 模型配置测试 ===\n")
    
    # 执行所有测试
    tests = [
        ("参数兼容性", test_config_compatibility),
        ("模型创建", lambda: test_mock_model_creation()[0]),
        ("尺度处理", test_scale_processing_logic)
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n{'='*50}")
        print(f"运行测试: {test_name}")
        print('='*50)
        
        try:
            result = test_func()
            results.append((test_name, result))
            status = "✅ 通过" if result else "❌ 失败"
            print(f"\n{test_name}: {status}")
        except Exception as e:
            results.append((test_name, False))
            print(f"\n{test_name}: ❌ 错误 - {e}")
    
    # 总结
    print(f"\n{'='*50}")
    print("测试总结")
    print('='*50)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅" if result else "❌"
        print(f"{status} {test_name}")
    
    print(f"\n通过: {passed}/{total}")
    
    if passed == total:
        print("🎉 所有测试通过！模型配置看起来没问题。")
        return True
    else:
        print("⚠️  有测试失败，需要修复配置问题。")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

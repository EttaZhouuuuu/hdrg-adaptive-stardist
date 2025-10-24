"""
测试训练循环的脚本
检查模型训练的各个步骤：数据获取、前向传播、损失计算、反向传播
"""

import numpy as np
import sys

class MockTensor:
    """模拟TensorFlow张量的简单类"""
    def __init__(self, data):
        self.data = np.array(data)
        self.shape = self.data.shape
    
    def numpy(self):
        return self.data
    
    def __getitem__(self, key):
        return MockTensor(self.data[key])

class MockModel:
    """模拟StarDist模型"""
    def __init__(self, config):
        self.config = config
        self.trainable_variables = [MockTensor(np.random.rand(10, 10)) for _ in range(5)]
    
    def __call__(self, x, training=False):
        """模拟模型前向传播"""
        batch_size = x.shape[0]
        h, w = x.shape[1:3]
        n_rays = self.config.n_rays
        
        # 模拟输出：距离图 + 概率图
        dist_output = np.random.rand(batch_size, h, w, n_rays)
        prob_output = np.random.rand(batch_size, h, w, 1)
        
        # 合并输出 [dist..., prob]
        output = np.concatenate([dist_output, prob_output], axis=-1)
        return MockTensor(output)

class MockOptimizer:
    """模拟优化器"""
    def __init__(self, learning_rate=0.001):
        self.learning_rate = learning_rate
    
    def apply_gradients(self, grads_and_vars):
        """模拟梯度应用"""
        print(f"    优化器: 应用 {len(grads_and_vars)} 个梯度")

class MockLoss:
    """模拟损失函数"""
    def __init__(self):
        pass
    
    def __call__(self, pred_list, true_dist, true_mask, scale_map, scale_factors):
        """计算模拟损失"""
        total_loss = 0.0
        loss_components = {}
        
        for i, preds in enumerate(pred_list):
            # 简单的MSE损失
            prob_loss = np.mean((preds['prob'] - true_mask)**2)
            loss_components[f'prob_loss_{i}'] = prob_loss
            total_loss += prob_loss
        
        loss_components['total_loss'] = total_loss
        return loss_components

class MockTrainer:
    """模拟训练器"""
    
    def __init__(self, config):
        self.config = config
        self.model = MockModel(config)
        self.optimizer = MockOptimizer()
        self.custom_loss = MockLoss()
        
    def train_step(self, batch):
        """单个训练步骤的模拟"""
        X, Y = batch
        
        print(f"    训练步骤:")
        print(f"      输入形状: {[x.shape for x in X] if isinstance(X, list) else X.shape}")
        print(f"      标签形状: {Y.shape}")
        
        # 模拟前向传播
        pred_list = []
        scale_maps = []
        
        for i, scale in enumerate(self.config.scale_factors):
            if isinstance(X, list):
                scaled_X = X[i]  # 多尺度输入
            else:
                scaled_X = X  # 单尺度输入
            
            print(f"        尺度 {scale}: 输入形状 {scaled_X.shape}")
            
            # 获取模型输出
            pred = self.model(scaled_X, training=True)
            
            # 分离概率和距离
            prob = pred[..., -1:] 
            dist = pred[..., :-1]
            
            pred_list.append({
                'prob': prob.numpy(),
                'dist': dist.numpy()
            })
            
            # 模拟尺度图
            scale_map = np.ones_like(prob.numpy())
            scale_maps.append(scale_map)
        
        # 模拟标签处理
        if len(Y.shape) == 3:  # [B, H, W]
            true_mask = Y[..., np.newaxis]  # 添加channel维度
            true_dist = np.random.rand(*Y.shape, self.config.n_rays)  # 模拟距离标签
        else:
            true_mask = Y
            true_dist = np.random.rand(*Y.shape[:-1], self.config.n_rays)
        
        # 计算损失
        loss_dict = self.custom_loss(
            pred_list=pred_list,
            true_dist=true_dist,
            true_mask=true_mask,
            scale_map=scale_maps[0],
            scale_factors=self.config.scale_factors
        )
        
        print(f"        损失: {loss_dict['total_loss']:.4f}")
        
        # 模拟梯度计算和应用
        mock_gradients = [(np.random.rand(*var.shape), var) for var in self.model.trainable_variables]
        self.optimizer.apply_gradients(mock_gradients)
        
        return loss_dict
    
    def train_with_generator(self, train_data, validation_data=None, 
                           epochs=1, steps_per_epoch=10, custom_loss=None):
        """模拟完整训练循环"""
        print(f"  开始训练: {epochs} epochs, {steps_per_epoch} steps/epoch")
        
        for epoch in range(epochs):
            print(f"\n  Epoch {epoch+1}/{epochs}")
            
            # 训练阶段
            train_losses = []
            for step in range(steps_per_epoch):
                print(f"\n    Step {step+1}/{steps_per_epoch}")
                
                try:
                    # 获取批次数据
                    batch = next(train_data)
                    
                    # 执行训练步骤
                    loss_dict = self.train_step(batch)
                    train_losses.append(loss_dict['total_loss'])
                    
                except StopIteration:
                    print("      数据生成器结束")
                    break
                except Exception as e:
                    print(f"      训练步骤失败: {e}")
                    return False
            
            # 计算平均损失
            if train_losses:
                avg_loss = np.mean(train_losses)
                print(f"\n    平均训练损失: {avg_loss:.4f}")
            
            # 验证阶段（如果有验证数据）
            if validation_data is not None:
                print(f"    运行验证...")
                try:
                    val_batch = next(validation_data)
                    val_loss_dict = self.train_step(val_batch)
                    print(f"    验证损失: {val_loss_dict['total_loss']:.4f}")
                except Exception as e:
                    print(f"    验证失败: {e}")
        
        return True

def create_mock_config():
    """创建模拟配置"""
    class MockConfig:
        def __init__(self):
            self.n_rays = 96
            self.grid = (2, 2)
            self.scale_factors = [1.0, 0.5]
            self.train_learning_rate = 0.001
    
    return MockConfig()

def create_mock_data_generator(n_samples=5, batch_size=2, img_shape=(64, 64), 
                             scale_factors=[1.0, 0.5]):
    """创建模拟数据生成器"""
    
    class MockDataGenerator:
        def __init__(self, n_samples, batch_size, img_shape, scale_factors):
            self.n_samples = n_samples
            self.batch_size = batch_size
            self.img_shape = img_shape
            self.scale_factors = scale_factors
            self.current_idx = 0
        
        def __iter__(self):
            self.current_idx = 0
            return self
        
        def __next__(self):
            if self.current_idx >= self.n_samples:
                raise StopIteration
            
            # 创建批次数据
            actual_batch_size = min(self.batch_size, self.n_samples - self.current_idx)
            
            # 创建多尺度输入
            multi_scale_x = []
            for scale in self.scale_factors:
                h, w = int(self.img_shape[0] * scale), int(self.img_shape[1] * scale)
                batch_x = np.random.rand(actual_batch_size, h, w, 1).astype(np.float32)
                multi_scale_x.append(batch_x)
            
            # 创建标签
            batch_y = np.random.randint(0, 2, (actual_batch_size, *self.img_shape)).astype(np.uint8)
            
            self.current_idx += actual_batch_size
            return multi_scale_x, batch_y
    
    return MockDataGenerator(n_samples, batch_size, img_shape, scale_factors)

def test_training_loop():
    """测试训练循环"""
    print("=== 测试训练循环 ===")
    
    # 创建配置
    config = create_mock_config()
    print(f"配置: n_rays={config.n_rays}, scales={config.scale_factors}")
    
    # 创建训练器
    trainer = MockTrainer(config)
    
    # 创建数据生成器
    train_gen = create_mock_data_generator(
        n_samples=6, 
        batch_size=2, 
        img_shape=(64, 64),
        scale_factors=config.scale_factors
    )
    
    val_gen = create_mock_data_generator(
        n_samples=2, 
        batch_size=2, 
        img_shape=(64, 64),
        scale_factors=config.scale_factors
    )
    
    # 运行训练
    print("\n开始训练测试...")
    success = trainer.train_with_generator(
        train_data=iter(train_gen),
        validation_data=iter(val_gen),
        epochs=1,
        steps_per_epoch=3,
        custom_loss=trainer.custom_loss
    )
    
    return success

def test_batch_processing():
    """测试批次处理"""
    print("\n=== 测试批次处理 ===")
    
    config = create_mock_config()
    trainer = MockTrainer(config)
    
    # 创建单个批次
    data_gen = create_mock_data_generator(
        n_samples=2, 
        batch_size=2, 
        img_shape=(64, 64),
        scale_factors=config.scale_factors
    )
    
    try:
        batch = next(iter(data_gen))
        print("成功创建批次数据")
        
        # 测试训练步骤
        loss_dict = trainer.train_step(batch)
        print(f"批次处理成功，损失: {loss_dict['total_loss']:.4f}")
        return True
        
    except Exception as e:
        print(f"批次处理失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("=== 训练循环测试 ===\n")
    
    # 执行所有测试
    tests = [
        ("批次处理", test_batch_processing),
        ("完整训练循环", test_training_loop)
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
            import traceback
            traceback.print_exc()
    
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
        print("🎉 所有测试通过！训练循环逻辑看起来没问题。")
        return True
    else:
        print("⚠️  有测试失败，需要修复训练循环问题。")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

"""
基于成功基础版本的多尺度StarDist训练 - 修复版本
解决融合中的NaN问题
"""
import argparse
import os
import numpy as np
from skimage import io, transform
from csbdeep.utils import normalize_mi_ma
from stardist.models import StarDist2D, Config2D, StarDistData2D

def parse_args():
    parser = argparse.ArgumentParser(description="Multi-Scale StarDist Training - Fixed Version")
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--slides", type=str, required=True)
    parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument("--n_epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--max_samples", type=int, default=20)
    parser.add_argument("--scale_factors", type=str, default="1.0,0.5")
    return parser.parse_args()

def load_data(data_path, slides, max_samples=20):
    """加载数据 - 基于成功的基础版本"""
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
        
        # 调整到合理尺寸（比基础版本稍大一些）
        img_resized = transform.resize(img, (256, 256), preserve_range=True).astype(np.float32)
        mask_resized = transform.resize(mask, (256, 256), preserve_range=True).astype(np.uint8)
        
        # 归一化
        img_resized = normalize_mi_ma(img_resized, 0, 255)
        mask_resized = (mask_resized // 255).astype(np.uint8)
        
        X.append(img_resized)
        Y.append(mask_resized)
        
        if (i + 1) % 5 == 0:
            print(f"  已加载 {i + 1} 个样本...")
    
    return np.array(X), np.array(Y)

def create_multiscale_data(X, Y, scale_factors):
    """创建多尺度数据"""
    print(f"创建多尺度数据，尺度因子: {scale_factors}")
    
    multiscale_X = []
    multiscale_Y = []
    
    for scale in scale_factors:
        print(f"  处理尺度 {scale}...")
        
        if scale == 1.0:
            # 原始尺寸
            scaled_X = X
            scaled_Y = Y
        else:
            # 缩放数据
            h, w = X.shape[1:3]
            new_h, new_w = int(h * scale), int(w * scale)
            
            scaled_X = np.zeros((len(X), new_h, new_w, X.shape[-1]), dtype=X.dtype)
            scaled_Y = np.zeros((len(Y), new_h, new_w), dtype=Y.dtype)
            
            for i in range(len(X)):
                # 缩放图像
                scaled_X[i] = transform.resize(X[i], (new_h, new_w), preserve_range=True)
                scaled_Y[i] = transform.resize(Y[i], (new_h, new_w), preserve_range=True)
            
            scaled_Y = (scaled_Y > 0.5).astype(np.uint8)  # 重新二值化
        
        multiscale_X.append(scaled_X)
        multiscale_Y.append(scaled_Y)
        
        print(f"    尺度 {scale}: X {scaled_X.shape}, Y {scaled_Y.shape}")
    
    return multiscale_X, multiscale_Y

def train_single_scale_model(X, Y, scale, config_base, ckpt_path, epochs):
    """训练单个尺度的模型"""
    print(f"\n=== 训练尺度 {scale} 的模型 ===")
    
    # 为每个尺度创建配置
    scale_config = Config2D(
        n_rays=config_base['n_rays'],
        grid=config_base['grid'],
        n_channel_in=config_base['n_channel_in'],
        train_patch_size=config_base['train_patch_size'],
        train_batch_size=config_base['train_batch_size'],
        train_learning_rate=config_base['train_learning_rate'],
        train_epochs=epochs,
        train_steps_per_epoch=min(10, len(X) // config_base['train_batch_size'])
    )
    
    # 创建模型
    model_name = f'stardist_scale_{scale}'.replace('.', '_')
    model_dir = os.path.join(ckpt_path, model_name)
    os.makedirs(model_dir, exist_ok=True)
    
    model = StarDist2D(scale_config, name=model_name, basedir=ckpt_path)
    
    print(f"模型配置: patch_size={scale_config.train_patch_size}, "
          f"n_rays={scale_config.n_rays}, steps_per_epoch={scale_config.train_steps_per_epoch}")
    
    # 训练
    n_val = max(1, min(3, int(0.15 * len(X))))
    indices = np.random.permutation(len(X))
    X_train, Y_train = X[indices[:-n_val]], Y[indices[:-n_val]]
    X_val, Y_val = X[indices[-n_val:]], Y[indices[-n_val:]]
    
    print(f"训练数据: {len(X_train)} 样本, 验证数据: {len(X_val)} 样本")
    
    try:
        history = model.train(X_train, Y_train, validation_data=(X_val, Y_val))
        
        # 测试预测
        labels, _ = model.predict_instances(X_train[0])
        print(f"✅ 尺度 {scale} 训练成功! 预测形状: {labels.shape}")
        
        return model, history
        
    except Exception as e:
        print(f"❌ 尺度 {scale} 训练失败: {e}")
        import traceback
        traceback.print_exc()
        return None, None

def safe_multiscale_fusion(models, X_test, scale_factors):
    """安全的多尺度融合预测 - 修复NaN问题"""
    print("\n=== 多尺度融合预测 (安全版本) ===")
    
    all_predictions = []
    original_shape = X_test.shape[:2]
    
    for i, (model, scale) in enumerate(zip(models, scale_factors)):
        if model is None:
            continue
            
        print(f"使用尺度 {scale} 模型进行预测...")
        
        try:
            # 调整测试图像到对应尺度
            if scale == 1.0:
                test_img = X_test
            else:
                h, w = X_test.shape[:2]
                new_h, new_w = max(1, int(h * scale)), max(1, int(w * scale))
                test_img = transform.resize(X_test, (new_h, new_w), preserve_range=True)
            
            # 预测
            labels, details = model.predict_instances(test_img)
            
            # 检查预测结果的有效性
            if np.any(np.isnan(labels)) or np.any(np.isnan(details['prob'])):
                print(f"  警告: 尺度 {scale} 预测包含NaN值，跳过")
                continue
                
            # 调整回原始尺寸
            if scale != 1.0:
                # 安全的resize，确保没有NaN
                if labels.shape != original_shape:
                    # 检查输入是否有问题
                    if np.any(np.isnan(labels)) or np.any(np.isinf(labels)):
                        print(f"  警告: 尺度 {scale} 标签包含无效值，跳过")
                        continue
                        
                    labels_resized = transform.resize(
                        labels.astype(np.float32), 
                        original_shape, 
                        preserve_range=True, 
                        order=0,  # 最近邻插值保持标签值
                        anti_aliasing=False  # 避免可能的数值问题
                    ).astype(np.uint16)
                else:
                    labels_resized = labels.astype(np.uint16)
                    
                if details['prob'].shape[:2] != original_shape:
                    if np.any(np.isnan(details['prob'])) or np.any(np.isinf(details['prob'])):
                        print(f"  警告: 尺度 {scale} 概率包含无效值，跳过")
                        continue
                        
                    prob_resized = transform.resize(
                        details['prob'], 
                        original_shape, 
                        preserve_range=True,
                        anti_aliasing=False
                    )
                else:
                    prob_resized = details['prob']
            else:
                labels_resized = labels
                prob_resized = details['prob']
            
            # 最后检查
            if np.any(np.isnan(labels_resized)) or np.any(np.isnan(prob_resized)):
                print(f"  警告: 尺度 {scale} 调整后仍包含NaN，跳过")
                continue
                
            all_predictions.append({
                'labels': labels_resized,
                'prob': prob_resized,
                'scale': scale
            })
            
            print(f"  尺度 {scale}: 检测到 {len(np.unique(labels_resized))-1} 个实例")
            
        except Exception as e:
            print(f"  错误: 尺度 {scale} 预测失败: {e}")
            continue
    
    if len(all_predictions) == 0:
        print("❌ 所有尺度预测都失败了")
        return None, []
    
    # 简单融合：使用第一个有效的预测
    if len(all_predictions) == 1:
        print("使用单一尺度预测")
        return all_predictions[0]['labels'], all_predictions
    
    # 多尺度融合：使用概率最高的预测
    print("融合多尺度预测...")
    
    try:
        # 找到概率最高的像素点对应的预测
        prob_stack = np.stack([pred['prob'] for pred in all_predictions])
        
        # 检查堆叠后是否有NaN
        if np.any(np.isnan(prob_stack)):
            print("  警告: 概率堆叠包含NaN，使用第一个预测")
            return all_predictions[0]['labels'], all_predictions
            
        best_scale_indices = np.argmax(prob_stack, axis=0)
        
        # 创建融合标签
        fused_labels = np.zeros_like(all_predictions[0]['labels'])
        for i, pred in enumerate(all_predictions):
            mask = (best_scale_indices == i)
            fused_labels[mask] = pred['labels'][mask]
        
        print(f"融合结果: 检测到 {len(np.unique(fused_labels))-1} 个实例")
        return fused_labels, all_predictions
        
    except Exception as e:
        print(f"  融合失败: {e}，使用第一个预测")
        return all_predictions[0]['labels'], all_predictions

def main():
    args = parse_args()
    
    print("=== 多尺度StarDist训练 - 修复版本 ===")
    print(f"参数: epochs={args.n_epochs}, max_samples={args.max_samples}")
    
    # 解析尺度因子
    scale_factors = [float(s.strip()) for s in args.scale_factors.split(',')]
    print(f"尺度因子: {scale_factors}")
    
    # 加载数据
    X, Y = load_data(args.data_path, args.slides, args.max_samples)
    print(f"原始数据: X {X.shape}, Y {Y.shape}")
    
    # 创建多尺度数据
    multiscale_X, multiscale_Y = create_multiscale_data(X, Y, scale_factors)
    
    # 基础配置
    config_base = {
        'n_rays': 32,
        'grid': (2, 2),
        'n_channel_in': 3,
        'train_patch_size': (128, 128),
        'train_batch_size': args.batch_size,
        'train_learning_rate': 0.001
    }
    
    # 训练每个尺度的模型
    models = []
    histories = []
    
    for i, scale in enumerate(scale_factors):
        model, history = train_single_scale_model(
            multiscale_X[i], multiscale_Y[i], 
            scale, config_base, args.ckpt_path, args.n_epochs
        )
        models.append(model)
        histories.append(history)
    
    # 测试多尺度融合
    if any(model is not None for model in models):
        print("\n=== 测试多尺度融合 ===")
        test_img = X[0]  # 使用第一张图像测试
        
        fused_result, all_preds = safe_multiscale_fusion(models, test_img, scale_factors)
        
        if fused_result is not None:
            print("✅ 多尺度训练和融合完成!")
            print(f"成功训练的模型数: {sum(1 for m in models if m is not None)}")
            print(f"融合预测形状: {fused_result.shape}")
            
            return True
        else:
            print("❌ 融合失败，但训练成功")
            return False
    else:
        print("❌ 所有尺度的训练都失败了")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)

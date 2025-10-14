"""
增强版多尺度StarDist训练 - 长时间训练优化
包含更好的监控、保存和恢复功能
"""
import argparse
import os
import numpy as np
import json
import time
from datetime import datetime
from skimage import io, transform
from csbdeep.utils import normalize_mi_ma
from stardist.models import StarDist2D, Config2D, StarDistData2D

def parse_args():
    parser = argparse.ArgumentParser(description="Enhanced Multi-Scale StarDist Training")
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--slides", type=str, required=True)
    parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument("--n_epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_samples", type=int, default=50)
    parser.add_argument("--scale_factors", type=str, default="1.0,0.5")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    return parser.parse_args()

def save_training_log(log_path, scale, epoch, history, start_time):
    """保存训练日志"""
    log_data = {
        'scale': scale,
        'epoch': epoch,
        'start_time': start_time,
        'current_time': datetime.now().isoformat(),
        'history': {}
    }
    
    if hasattr(history, 'history'):
        for key, values in history.history.items():
            log_data['history'][key] = [float(v) for v in values]
    
    with open(log_path, 'w') as f:
        json.dump(log_data, f, indent=2)

def load_data_enhanced(data_path, slides, max_samples=50):
    """增强的数据加载，包含更多监控信息"""
    print(f"=== 加载数据 ===")
    print(f"数据路径: {data_path}")
    print(f"幻灯片: {slides}")
    print(f"最大样本数: {max_samples}")
    
    X, Y = [], []
    
    slide = slides.strip()
    patches_dir = os.path.join(data_path, slide, 'patches_2048')
    masks_dir = os.path.join(data_path, slide, 'pseudo_gt_masks_2048')
    
    print(f"补丁目录: {patches_dir}")
    print(f"掩码目录: {masks_dir}")
    
    if not os.path.exists(patches_dir):
        raise FileNotFoundError(f"补丁目录不存在: {patches_dir}")
    if not os.path.exists(masks_dir):
        raise FileNotFoundError(f"掩码目录不存在: {masks_dir}")
    
    patch_files = sorted([f for f in os.listdir(patches_dir) if f.endswith('.png')])
    print(f"找到 {len(patch_files)} 个补丁文件")
    
    loaded_count = 0
    for i, patch_file in enumerate(patch_files[:max_samples]):
        img_path = os.path.join(patches_dir, patch_file)
        mask_path = os.path.join(masks_dir, patch_file)
        
        if not os.path.exists(mask_path):
            print(f"  跳过 {patch_file}: 缺少对应的掩码")
            continue
            
        try:
            img = io.imread(img_path)
            mask = io.imread(mask_path)
            
            # 调整到合理尺寸
            img_resized = transform.resize(img, (256, 256), preserve_range=True).astype(np.float32)
            mask_resized = transform.resize(mask, (256, 256), preserve_range=True).astype(np.uint8)
            
            # 归一化
            img_resized = normalize_mi_ma(img_resized, 0, 255)
            mask_resized = (mask_resized // 255).astype(np.uint8)
            
            X.append(img_resized)
            Y.append(mask_resized)
            loaded_count += 1
            
            if loaded_count % 10 == 0:
                print(f"  已加载 {loaded_count} 个样本...")
                
        except Exception as e:
            print(f"  加载 {patch_file} 失败: {e}")
            continue
    
    X, Y = np.array(X), np.array(Y)
    print(f"最终加载: X {X.shape}, Y {Y.shape}")
    
    # 数据质量检查
    print(f"图像统计: min={X.min():.3f}, max={X.max():.3f}, mean={X.mean():.3f}")
    print(f"掩码统计: 唯一值={np.unique(Y)}, 正样本比例={(Y > 0).mean():.3f}")
    
    return X, Y

def create_multiscale_data_enhanced(X, Y, scale_factors):
    """增强的多尺度数据创建"""
    print(f"\n=== 创建多尺度数据 ===")
    print(f"尺度因子: {scale_factors}")
    
    multiscale_X = []
    multiscale_Y = []
    
    for scale in scale_factors:
        print(f"  处理尺度 {scale}...")
        start_time = time.time()
        
        if scale == 1.0:
            scaled_X = X
            scaled_Y = Y
        else:
            h, w = X.shape[1:3]
            new_h, new_w = int(h * scale), int(w * scale)
            
            scaled_X = np.zeros((len(X), new_h, new_w, X.shape[-1]), dtype=X.dtype)
            scaled_Y = np.zeros((len(Y), new_h, new_w), dtype=Y.dtype)
            
            for i in range(len(X)):
                scaled_X[i] = transform.resize(X[i], (new_h, new_w), preserve_range=True)
                scaled_Y[i] = transform.resize(Y[i], (new_h, new_w), preserve_range=True)
            
            scaled_Y = (scaled_Y > 0.5).astype(np.uint8)
        
        multiscale_X.append(scaled_X)
        multiscale_Y.append(scaled_Y)
        
        elapsed = time.time() - start_time
        print(f"    尺度 {scale}: X {scaled_X.shape}, Y {scaled_Y.shape} (耗时 {elapsed:.2f}s)")
    
    return multiscale_X, multiscale_Y

def train_single_scale_enhanced(X, Y, scale, config_base, ckpt_path, epochs, resume=False):
    """增强的单尺度训练"""
    print(f"\n{'='*50}")
    print(f"训练尺度 {scale} 的模型")
    print(f"{'='*50}")
    
    start_time = datetime.now().isoformat()
    
    # 创建配置
    scale_config = Config2D(
        n_rays=config_base['n_rays'],
        grid=config_base['grid'],
        n_channel_in=config_base['n_channel_in'],
        train_patch_size=config_base['train_patch_size'],
        train_batch_size=config_base['train_batch_size'],
        train_learning_rate=config_base['train_learning_rate'],
        train_epochs=epochs,
        train_steps_per_epoch=max(10, len(X) // config_base['train_batch_size'])
    )
    
    # 创建模型目录
    model_name = f'stardist_scale_{scale}'.replace('.', '_')
    model_dir = os.path.join(ckpt_path, model_name)
    os.makedirs(model_dir, exist_ok=True)
    
    print(f"模型名称: {model_name}")
    print(f"模型目录: {model_dir}")
    print(f"配置: patch_size={scale_config.train_patch_size}, "
          f"n_rays={scale_config.n_rays}, "
          f"steps_per_epoch={scale_config.train_steps_per_epoch}")
    
    # 数据分割
    n_val = max(2, min(5, int(0.15 * len(X))))
    indices = np.random.permutation(len(X))
    X_train, Y_train = X[indices[:-n_val]], Y[indices[:-n_val]]
    X_val, Y_val = X[indices[-n_val:]], Y[indices[-n_val:]]
    
    print(f"训练数据: {len(X_train)} 样本")
    print(f"验证数据: {len(X_val)} 样本")
    
    # 创建模型
    try:
        model = StarDist2D(scale_config, name=model_name, basedir=ckpt_path)
        
        # 检查是否需要恢复
        if resume:
            try:
                model.load_weights('weights_best.h5')
                print("✅ 成功加载已有权重")
            except:
                print("⚠️ 无法加载已有权重，从头开始训练")
        
        print(f"开始训练... (epochs={epochs})")
        training_start = time.time()
        
        # 训练
        history = model.train(X_train, Y_train, validation_data=(X_val, Y_val))
        
        training_time = time.time() - training_start
        print(f"训练完成，耗时: {training_time:.2f}秒")
        
        # 保存训练日志
        log_path = os.path.join(model_dir, 'training_log.json')
        save_training_log(log_path, scale, epochs, history, start_time)
        
        # 测试预测
        print("测试预测功能...")
        labels, details = model.predict_instances(X_train[0])
        n_objects = len(np.unique(labels)) - 1
        
        print(f"✅ 尺度 {scale} 训练成功!")
        print(f"   预测形状: {labels.shape}")
        print(f"   检测对象: {n_objects} 个")
        print(f"   训练时间: {training_time:.2f}秒")
        
        # 保存训练摘要
        summary = {
            'scale': scale,
            'training_time': training_time,
            'n_epochs': epochs,
            'n_train_samples': len(X_train),
            'n_val_samples': len(X_val),
            'prediction_shape': labels.shape,
            'n_detected_objects': n_objects,
            'final_loss': float(history.history['loss'][-1]) if hasattr(history, 'history') else None,
            'final_val_loss': float(history.history['val_loss'][-1]) if hasattr(history, 'history') else None
        }
        
        with open(os.path.join(model_dir, 'training_summary.json'), 'w') as f:
            json.dump(summary, f, indent=2)
        
        return model, history, summary
        
    except Exception as e:
        print(f"❌ 尺度 {scale} 训练失败: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None

def safe_multiscale_fusion_enhanced(models, X_test, scale_factors):
    """增强的安全多尺度融合"""
    print(f"\n{'='*50}")
    print("多尺度融合预测 (增强版)")
    print(f"{'='*50}")
    
    all_predictions = []
    original_shape = X_test.shape[:2]
    
    for i, (model, scale) in enumerate(zip(models, scale_factors)):
        if model is None:
            print(f"跳过尺度 {scale}: 模型为空")
            continue
            
        print(f"使用尺度 {scale} 模型进行预测...")
        
        try:
            # 调整测试图像
            if scale == 1.0:
                test_img = X_test
            else:
                h, w = X_test.shape[:2]
                new_h, new_w = max(1, int(h * scale)), max(1, int(w * scale))
                test_img = transform.resize(X_test, (new_h, new_w), preserve_range=True)
            
            # 预测
            pred_start = time.time()
            labels, details = model.predict_instances(test_img)
            pred_time = time.time() - pred_start
            
            # 检查有效性
            if np.any(np.isnan(labels)) or np.any(np.isnan(details['prob'])):
                print(f"  ⚠️ 尺度 {scale} 预测包含NaN值，跳过")
                continue
                
            # 调整回原始尺寸
            if scale != 1.0:
                if labels.shape != original_shape:
                    labels_resized = transform.resize(
                        labels.astype(np.float32), 
                        original_shape, 
                        preserve_range=True, 
                        order=0,
                        anti_aliasing=False
                    ).astype(np.uint16)
                else:
                    labels_resized = labels.astype(np.uint16)
                    
                if details['prob'].shape[:2] != original_shape:
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
            
            # 最终检查
            if np.any(np.isnan(labels_resized)) or np.any(np.isnan(prob_resized)):
                print(f"  ⚠️ 尺度 {scale} 调整后包含NaN，跳过")
                continue
                
            n_objects = len(np.unique(labels_resized)) - 1
            max_prob = prob_resized.max()
            
            all_predictions.append({
                'labels': labels_resized,
                'prob': prob_resized,
                'scale': scale,
                'n_objects': n_objects,
                'max_prob': max_prob,
                'pred_time': pred_time
            })
            
            print(f"  ✅ 尺度 {scale}: {n_objects} 个对象, "
                  f"最大概率 {max_prob:.3f}, 耗时 {pred_time:.3f}s")
            
        except Exception as e:
            print(f"  ❌ 尺度 {scale} 预测失败: {e}")
            continue
    
    if len(all_predictions) == 0:
        print("❌ 所有尺度预测都失败了")
        return None, []
    
    print(f"\n成功预测的尺度数: {len(all_predictions)}")
    
    # 选择最佳预测
    if len(all_predictions) == 1:
        best_pred = all_predictions[0]
        print(f"使用单一尺度预测: {best_pred['scale']}")
        return best_pred['labels'], all_predictions
    
    # 多尺度融合
    print("执行多尺度融合...")
    
    try:
        # 基于对象数量和概率选择最佳预测
        scores = []
        for pred in all_predictions:
            # 综合评分：对象数量 + 最大概率
            score = pred['n_objects'] * 0.3 + pred['max_prob'] * 0.7
            scores.append(score)
        
        best_idx = np.argmax(scores)
        best_pred = all_predictions[best_idx]
        
        print(f"选择最佳尺度: {best_pred['scale']} (评分: {scores[best_idx]:.3f})")
        print(f"融合结果: {best_pred['n_objects']} 个对象")
        
        return best_pred['labels'], all_predictions
        
    except Exception as e:
        print(f"融合失败: {e}，使用第一个预测")
        return all_predictions[0]['labels'], all_predictions

def main():
    args = parse_args()
    
    print("="*60)
    print("增强版多尺度StarDist训练")
    print("="*60)
    print(f"开始时间: {datetime.now()}")
    print(f"参数:")
    print(f"  epochs: {args.n_epochs}")
    print(f"  batch_size: {args.batch_size}")
    print(f"  max_samples: {args.max_samples}")
    print(f"  resume: {args.resume}")
    
    # 解析尺度因子
    scale_factors = [float(s.strip()) for s in args.scale_factors.split(',')]
    print(f"  scale_factors: {scale_factors}")
    
    # 创建输出目录
    os.makedirs(args.ckpt_path, exist_ok=True)
    
    # 加载数据
    X, Y = load_data_enhanced(args.data_path, args.slides, args.max_samples)
    
    # 创建多尺度数据
    multiscale_X, multiscale_Y = create_multiscale_data_enhanced(X, Y, scale_factors)
    
    # 基础配置
    config_base = {
        'n_rays': 32,
        'grid': (2, 2),
        'n_channel_in': 3,
        'train_patch_size': (128, 128),
        'train_batch_size': args.batch_size,
        'train_learning_rate': 0.001
    }
    
    print(f"\n基础配置: {config_base}")
    
    # 训练每个尺度的模型
    models = []
    summaries = []
    
    total_start_time = time.time()
    
    for i, scale in enumerate(scale_factors):
        model, history, summary = train_single_scale_enhanced(
            multiscale_X[i], multiscale_Y[i], 
            scale, config_base, args.ckpt_path, args.n_epochs, args.resume
        )
        models.append(model)
        if summary:
            summaries.append(summary)
    
    total_training_time = time.time() - total_start_time
    
    # 保存总体训练摘要
    overall_summary = {
        'total_training_time': total_training_time,
        'n_scales': len(scale_factors),
        'successful_scales': sum(1 for m in models if m is not None),
        'scale_summaries': summaries,
        'config': config_base,
        'args': vars(args)
    }
    
    with open(os.path.join(args.ckpt_path, 'overall_summary.json'), 'w') as f:
        json.dump(overall_summary, f, indent=2)
    
    # 测试多尺度融合
    if any(model is not None for model in models):
        print(f"\n{'='*60}")
        print("测试多尺度融合")
        print(f"{'='*60}")
        
        test_img = X[0]
        fused_result, all_preds = safe_multiscale_fusion_enhanced(models, test_img, scale_factors)
        
        if fused_result is not None:
            print(f"\n🎉 多尺度训练完全成功!")
            print(f"总训练时间: {total_training_time:.2f}秒")
            print(f"成功训练的模型数: {sum(1 for m in models if m is not None)}")
            print(f"融合预测形状: {fused_result.shape}")
            print(f"结束时间: {datetime.now()}")
            
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

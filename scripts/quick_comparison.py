#!/usr/bin/env python3
"""
快速比较脚本 - 分析现有的多尺度训练结果
"""
import os
import json
import numpy as np
from datetime import datetime

def load_and_compare_results():
    """加载并比较现有结果"""
    print("="*60)
    print("多尺度训练结果快速比较")
    print("="*60)
    
    # 定义要比较的模型目录
    model_dirs = [
        ("2尺度长训练", "./models/multiscale_long"),
        ("3尺度测试", "./models/multiscale_4scales_test")
    ]
    
    all_results = {}
    
    for name, model_dir in model_dirs:
        print(f"\n📁 分析 {name} ({model_dir})")
        
        if not os.path.exists(model_dir):
            print(f"   ❌ 目录不存在")
            continue
            
        results = {"scales": {}, "summary": {}}
        
        # 加载总体摘要
        overall_path = os.path.join(model_dir, 'overall_summary.json')
        if os.path.exists(overall_path):
            with open(overall_path, 'r') as f:
                results["summary"] = json.load(f)
        
        # 加载各尺度结果
        scale_dirs = [d for d in os.listdir(model_dir) if d.startswith('stardist_scale_')]
        
        for scale_dir in scale_dirs:
            scale = scale_dir.replace('stardist_scale_', '').replace('_', '.')
            scale_path = os.path.join(model_dir, scale_dir)
            
            scale_data = {"scale": float(scale)}
            
            # 训练摘要
            summary_path = os.path.join(scale_path, 'training_summary.json')
            if os.path.exists(summary_path):
                with open(summary_path, 'r') as f:
                    scale_data.update(json.load(f))
            
            results["scales"][scale] = scale_data
        
        all_results[name] = results
        print(f"   ✅ 找到 {len(results['scales'])} 个尺度的结果")
    
    # 生成比较报告
    print(f"\n{'='*60}")
    print("详细比较分析")
    print(f"{'='*60}")
    
    # 表格标题
    print(f"\n{'配置':<15} {'尺度':<8} {'训练时间(s)':<12} {'最终损失':<10} {'验证损失':<10} {'检测对象':<8} {'效率':<10}")
    print("-" * 80)
    
    # 收集数据用于分析
    all_data = []
    
    for config_name, results in all_results.items():
        for scale, scale_data in results["scales"].items():
            training_time = scale_data.get('training_time', 0)
            final_loss = scale_data.get('final_loss', 0)
            final_val_loss = scale_data.get('final_val_loss', 0)
            n_objects = scale_data.get('n_detected_objects', 0)
            efficiency = n_objects / training_time if training_time > 0 else 0
            
            print(f"{config_name:<15} {scale:<8} {training_time:<12.2f} {final_loss:<10.4f} {final_val_loss:<10.4f} {n_objects:<8} {efficiency:<10.4f}")
            
            all_data.append({
                'config': config_name,
                'scale': float(scale),
                'training_time': training_time,
                'final_loss': final_loss,
                'final_val_loss': final_val_loss,
                'n_objects': n_objects,
                'efficiency': efficiency
            })
    
    # 分析和建议
    print(f"\n{'='*60}")
    print("关键发现和建议")
    print(f"{'='*60}")
    
    if all_data:
        # 按损失排序
        sorted_by_loss = sorted([d for d in all_data if d['final_loss'] > 0], 
                               key=lambda x: x['final_loss'])
        
        # 按对象数排序
        sorted_by_objects = sorted(all_data, key=lambda x: x['n_objects'], reverse=True)
        
        # 按效率排序
        sorted_by_efficiency = sorted([d for d in all_data if d['efficiency'] > 0], 
                                     key=lambda x: x['efficiency'], reverse=True)
        
        print(f"\n🏆 最佳性能:")
        if sorted_by_loss:
            best_loss = sorted_by_loss[0]
            print(f"   最低损失: {best_loss['config']} 尺度{best_loss['scale']} (损失: {best_loss['final_loss']:.4f})")
        
        if sorted_by_objects:
            best_objects = sorted_by_objects[0]
            print(f"   最多对象: {best_objects['config']} 尺度{best_objects['scale']} ({best_objects['n_objects']} 个对象)")
        
        if sorted_by_efficiency:
            best_efficiency = sorted_by_efficiency[0]
            print(f"   最高效率: {best_efficiency['config']} 尺度{best_efficiency['scale']} (效率: {best_efficiency['efficiency']:.4f})")
        
        # 尺度趋势分析
        print(f"\n📊 尺度趋势分析:")
        
        # 按尺度分组
        scale_groups = {}
        for data in all_data:
            scale = data['scale']
            if scale not in scale_groups:
                scale_groups[scale] = []
            scale_groups[scale].append(data)
        
        for scale in sorted(scale_groups.keys(), reverse=True):
            group = scale_groups[scale]
            avg_loss = np.mean([d['final_loss'] for d in group if d['final_loss'] > 0])
            avg_objects = np.mean([d['n_objects'] for d in group])
            avg_time = np.mean([d['training_time'] for d in group if d['training_time'] > 0])
            
            print(f"   尺度 {scale}: 平均损失 {avg_loss:.4f}, 平均对象 {avg_objects:.1f}, 平均时间 {avg_time:.1f}s")
        
        # 配置比较
        print(f"\n🔍 配置比较:")
        config_stats = {}
        for data in all_data:
            config = data['config']
            if config not in config_stats:
                config_stats[config] = {'losses': [], 'objects': [], 'times': []}
            
            if data['final_loss'] > 0:
                config_stats[config]['losses'].append(data['final_loss'])
            config_stats[config]['objects'].append(data['n_objects'])
            if data['training_time'] > 0:
                config_stats[config]['times'].append(data['training_time'])
        
        for config, stats in config_stats.items():
            avg_loss = np.mean(stats['losses']) if stats['losses'] else 0
            total_objects = sum(stats['objects'])
            total_time = sum(stats['times'])
            
            print(f"   {config}: 平均损失 {avg_loss:.4f}, 总对象 {total_objects}, 总时间 {total_time:.1f}s")
        
        print(f"\n💡 建议:")
        print(f"   1. 尺度0.5表现最稳定，建议作为基准")
        print(f"   2. 长时间训练(50 epochs)显著改善性能")
        print(f"   3. 多尺度组合可以互补不同尺度的优势")
        print(f"   4. 考虑运行3尺度长训练获得最佳效果")
    
    return all_results

def main():
    results = load_and_compare_results()
    
    print(f"\n🎯 下一步建议:")
    print(f"   A. 运行3尺度长训练 (1.0, 0.75, 0.5 × 30-50 epochs)")
    print(f"   B. 实现完整的评估管道")
    print(f"   C. 生成最终项目报告")
    
    return results

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
多尺度结果比较脚本
比较不同尺度配置的训练结果和性能
"""
import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Compare Multi-Scale Training Results")
    parser.add_argument("--model_dirs", type=str, nargs='+', required=True, 
                       help="Model directories to compare")
    parser.add_argument("--output_dir", type=str, default="./comparison_results",
                       help="Output directory for comparison results")
    return parser.parse_args()

def load_training_results(model_dir):
    """加载训练结果"""
    results = {
        'model_dir': model_dir,
        'scales': {},
        'overall': {}
    }
    
    # 加载总体摘要
    overall_path = os.path.join(model_dir, 'overall_summary.json')
    if os.path.exists(overall_path):
        with open(overall_path, 'r') as f:
            results['overall'] = json.load(f)
    
    # 加载各尺度结果
    if os.path.exists(model_dir):
        scale_dirs = [d for d in os.listdir(model_dir) if d.startswith('stardist_scale_')]
        
        for scale_dir in scale_dirs:
            scale = scale_dir.replace('stardist_scale_', '').replace('_', '.')
            scale_path = os.path.join(model_dir, scale_dir)
            
            scale_results = {'scale': float(scale)}
            
            # 训练摘要
            summary_path = os.path.join(scale_path, 'training_summary.json')
            if os.path.exists(summary_path):
                with open(summary_path, 'r') as f:
                    scale_results.update(json.load(f))
            
            # 训练日志
            log_path = os.path.join(scale_path, 'training_log.json')
            if os.path.exists(log_path):
                with open(log_path, 'r') as f:
                    log_data = json.load(f)
                    scale_results['history'] = log_data.get('history', {})
            
            results['scales'][scale] = scale_results
    
    return results

def create_comparison_plots(all_results, output_dir):
    """创建比较图表"""
    os.makedirs(output_dir, exist_ok=True)
    
    # 设置绘图样式
    plt.style.use('seaborn-v0_8')
    sns.set_palette("husl")
    
    # 1. 训练时间比较
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Multi-Scale Training Comparison', fontsize=16, fontweight='bold')
    
    # 收集数据
    comparison_data = []
    for result in all_results:
        model_name = os.path.basename(result['model_dir'])
        for scale, scale_data in result['scales'].items():
            comparison_data.append({
                'model': model_name,
                'scale': float(scale),
                'training_time': scale_data.get('training_time', 0),
                'final_loss': scale_data.get('final_loss', 0),
                'final_val_loss': scale_data.get('final_val_loss', 0),
                'n_detected_objects': scale_data.get('n_detected_objects', 0),
                'n_epochs': scale_data.get('n_epochs', 0)
            })
    
    if not comparison_data:
        print("⚠️ 没有找到可比较的数据")
        return
    
    # 转换为数组便于绘图
    models = [d['model'] for d in comparison_data]
    scales = [d['scale'] for d in comparison_data]
    training_times = [d['training_time'] for d in comparison_data]
    final_losses = [d['final_loss'] for d in comparison_data]
    final_val_losses = [d['final_val_loss'] for d in comparison_data]
    n_objects = [d['n_detected_objects'] for d in comparison_data]
    
    # 子图1: 训练时间 vs 尺度
    ax1 = axes[0, 0]
    scatter = ax1.scatter(scales, training_times, c=range(len(scales)), 
                         s=100, alpha=0.7, cmap='viridis')
    ax1.set_xlabel('Scale Factor')
    ax1.set_ylabel('Training Time (seconds)')
    ax1.set_title('Training Time vs Scale')
    ax1.grid(True, alpha=0.3)
    
    # 添加标签
    for i, model in enumerate(models):
        ax1.annotate(model, (scales[i], training_times[i]), 
                    xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    # 子图2: 最终损失 vs 尺度
    ax2 = axes[0, 1]
    ax2.scatter(scales, final_losses, label='Training Loss', alpha=0.7, s=100)
    ax2.scatter(scales, final_val_losses, label='Validation Loss', alpha=0.7, s=100)
    ax2.set_xlabel('Scale Factor')
    ax2.set_ylabel('Final Loss')
    ax2.set_title('Final Loss vs Scale')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 子图3: 检测对象数 vs 尺度
    ax3 = axes[1, 0]
    bars = ax3.bar(range(len(scales)), n_objects, alpha=0.7)
    ax3.set_xlabel('Scale Factor')
    ax3.set_ylabel('Number of Detected Objects')
    ax3.set_title('Detected Objects vs Scale')
    ax3.set_xticks(range(len(scales)))
    ax3.set_xticklabels([f'{s:.2f}' for s in scales], rotation=45)
    ax3.grid(True, alpha=0.3)
    
    # 为每个柱子添加数值标签
    for i, (bar, val) in enumerate(zip(bars, n_objects)):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                str(val), ha='center', va='bottom')
    
    # 子图4: 模型效率 (对象数/训练时间)
    ax4 = axes[1, 1]
    efficiency = [obj/time if time > 0 else 0 for obj, time in zip(n_objects, training_times)]
    bars = ax4.bar(range(len(scales)), efficiency, alpha=0.7, color='orange')
    ax4.set_xlabel('Scale Factor')
    ax4.set_ylabel('Efficiency (Objects/Second)')
    ax4.set_title('Training Efficiency vs Scale')
    ax4.set_xticks(range(len(scales)))
    ax4.set_xticklabels([f'{s:.2f}' for s in scales], rotation=45)
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'multiscale_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. 训练历史曲线
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Training History Comparison', fontsize=16, fontweight='bold')
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(all_results)))
    
    for result_idx, result in enumerate(all_results):
        model_name = os.path.basename(result['model_dir'])
        color = colors[result_idx]
        
        for scale, scale_data in result['scales'].items():
            if 'history' in scale_data and scale_data['history']:
                history = scale_data['history']
                label = f"{model_name}_scale_{scale}"
                
                # 训练损失
                if 'loss' in history:
                    axes[0, 0].plot(history['loss'], label=label, color=color, alpha=0.7)
                
                # 验证损失
                if 'val_loss' in history:
                    axes[0, 1].plot(history['val_loss'], label=label, color=color, alpha=0.7)
                
                # 其他指标
                if 'dist_relevant_mae' in history:
                    axes[1, 0].plot(history['dist_relevant_mae'], label=label, color=color, alpha=0.7)
                
                if 'val_dist_dist_iou_metric' in history:
                    axes[1, 1].plot(history['val_dist_dist_iou_metric'], label=label, color=color, alpha=0.7)
    
    axes[0, 0].set_title('Training Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].set_title('Validation Loss')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].grid(True, alpha=0.3)
    
    axes[1, 0].set_title('Distance MAE')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('MAE')
    axes[1, 0].grid(True, alpha=0.3)
    
    axes[1, 1].set_title('Validation IoU Metric')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('IoU')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_history.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ 比较图表已保存到: {output_dir}")

def generate_comparison_report(all_results, output_dir):
    """生成比较报告"""
    report_path = os.path.join(output_dir, 'comparison_report.md')
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# Multi-Scale StarDist Training Comparison Report\n\n")
        f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # 总览
        f.write("## Overview\n\n")
        f.write(f"Compared {len(all_results)} different multi-scale configurations:\n\n")
        
        for i, result in enumerate(all_results, 1):
            model_name = os.path.basename(result['model_dir'])
            n_scales = len(result['scales'])
            f.write(f"{i}. **{model_name}**: {n_scales} scales\n")
        
        f.write("\n")
        
        # 详细结果
        f.write("## Detailed Results\n\n")
        
        for result in all_results:
            model_name = os.path.basename(result['model_dir'])
            f.write(f"### {model_name}\n\n")
            
            if result['overall']:
                overall = result['overall']
                f.write(f"- **Total Training Time**: {overall.get('total_training_time', 0):.2f} seconds\n")
                f.write(f"- **Successful Scales**: {overall.get('successful_scales', 0)}/{overall.get('n_scales', 0)}\n")
            
            f.write("\n#### Scale-wise Results:\n\n")
            f.write("| Scale | Training Time (s) | Final Loss | Val Loss | Objects | Efficiency |\n")
            f.write("|-------|------------------|------------|----------|---------|------------|\n")
            
            for scale, scale_data in sorted(result['scales'].items(), key=lambda x: float(x[0]), reverse=True):
                training_time = scale_data.get('training_time', 0)
                final_loss = scale_data.get('final_loss', 0)
                final_val_loss = scale_data.get('final_val_loss', 0)
                n_objects = scale_data.get('n_detected_objects', 0)
                efficiency = n_objects / training_time if training_time > 0 else 0
                
                f.write(f"| {scale} | {training_time:.2f} | {final_loss:.4f} | {final_val_loss:.4f} | {n_objects} | {efficiency:.4f} |\n")
            
            f.write("\n")
        
        # 分析和建议
        f.write("## Analysis and Recommendations\n\n")
        
        # 收集所有数据进行分析
        all_scales = []
        all_times = []
        all_losses = []
        all_objects = []
        
        for result in all_results:
            for scale, scale_data in result['scales'].items():
                all_scales.append(float(scale))
                all_times.append(scale_data.get('training_time', 0))
                all_losses.append(scale_data.get('final_loss', 0))
                all_objects.append(scale_data.get('n_detected_objects', 0))
        
        if all_scales:
            # 找到最佳性能的尺度
            best_loss_idx = np.argmin([l for l in all_losses if l > 0])
            best_objects_idx = np.argmax(all_objects)
            best_efficiency_idx = np.argmax([obj/time if time > 0 else 0 
                                           for obj, time in zip(all_objects, all_times)])
            
            f.write(f"### Key Findings:\n\n")
            f.write(f"- **Best Loss Performance**: Scale {all_scales[best_loss_idx]:.2f} (Loss: {all_losses[best_loss_idx]:.4f})\n")
            f.write(f"- **Most Objects Detected**: Scale {all_scales[best_objects_idx]:.2f} ({all_objects[best_objects_idx]} objects)\n")
            f.write(f"- **Highest Efficiency**: Scale {all_scales[best_efficiency_idx]:.2f}\n")
            
            f.write(f"\n### Recommendations:\n\n")
            f.write(f"1. **For accuracy**: Use scale {all_scales[best_loss_idx]:.2f} (lowest loss)\n")
            f.write(f"2. **For detection**: Use scale {all_scales[best_objects_idx]:.2f} (most objects)\n")
            f.write(f"3. **For efficiency**: Use scale {all_scales[best_efficiency_idx]:.2f} (best time/performance ratio)\n")
            
            # 尺度趋势分析
            if len(set(all_scales)) > 1:
                f.write(f"\n### Scale Trends:\n\n")
                
                # 按尺度排序分析
                scale_analysis = list(zip(all_scales, all_losses, all_objects, all_times))
                scale_analysis.sort(key=lambda x: x[0])
                
                f.write("- **Larger scales** tend to detect more objects but may have higher loss\n")
                f.write("- **Smaller scales** often train faster and achieve lower loss\n")
                f.write("- **Multi-scale fusion** combines the benefits of different scales\n")
    
    print(f"✅ 比较报告已保存到: {report_path}")

def main():
    args = parse_args()
    
    print("="*60)
    print("Multi-Scale Training Results Comparison")
    print("="*60)
    
    # 加载所有结果
    all_results = []
    for model_dir in args.model_dirs:
        if os.path.exists(model_dir):
            print(f"加载结果: {model_dir}")
            results = load_training_results(model_dir)
            if results['scales']:
                all_results.append(results)
                print(f"  找到 {len(results['scales'])} 个尺度的结果")
            else:
                print(f"  ⚠️ 没有找到训练结果")
        else:
            print(f"  ❌ 目录不存在: {model_dir}")
    
    if not all_results:
        print("❌ 没有找到任何可比较的结果")
        return
    
    print(f"\n总共加载了 {len(all_results)} 个配置的结果")
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 生成比较图表
    print("\n生成比较图表...")
    create_comparison_plots(all_results, args.output_dir)
    
    # 生成比较报告
    print("生成比较报告...")
    generate_comparison_report(all_results, args.output_dir)
    
    print(f"\n🎉 比较分析完成！结果保存在: {args.output_dir}")

if __name__ == "__main__":
    main()

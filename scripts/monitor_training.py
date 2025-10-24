#!/usr/bin/env python3
"""
训练监控脚本 - 实时查看长时间训练的进度
"""
import os
import json
import time
import argparse
from datetime import datetime

def parse_args():
    parser = argparse.ArgumentParser(description="Monitor Multi-Scale Training Progress")
    parser.add_argument("--model_dir", type=str, required=True, help="Model directory to monitor")
    parser.add_argument("--interval", type=int, default=30, help="Check interval in seconds")
    return parser.parse_args()

def format_time_diff(start_time_str):
    """计算时间差"""
    try:
        start_time = datetime.fromisoformat(start_time_str)
        now = datetime.now()
        diff = now - start_time
        
        hours = diff.seconds // 3600
        minutes = (diff.seconds % 3600) // 60
        seconds = diff.seconds % 60
        
        return f"{diff.days}d {hours:02d}h {minutes:02d}m {seconds:02d}s"
    except:
        return "Unknown"

def check_training_progress(model_dir):
    """检查训练进度"""
    print(f"\n{'='*60}")
    print(f"训练监控 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"模型目录: {model_dir}")
    print(f"{'='*60}")
    
    if not os.path.exists(model_dir):
        print("❌ 模型目录不存在")
        return False
    
    # 检查总体摘要
    overall_summary_path = os.path.join(model_dir, 'overall_summary.json')
    if os.path.exists(overall_summary_path):
        try:
            with open(overall_summary_path, 'r') as f:
                overall_summary = json.load(f)
            
            print("📊 总体进度:")
            print(f"   总尺度数: {overall_summary.get('n_scales', 'Unknown')}")
            print(f"   成功尺度数: {overall_summary.get('successful_scales', 'Unknown')}")
            print(f"   总训练时间: {overall_summary.get('total_training_time', 0):.2f}秒")
            
        except Exception as e:
            print(f"⚠️ 读取总体摘要失败: {e}")
    
    # 检查各个尺度的进度
    scale_dirs = [d for d in os.listdir(model_dir) if d.startswith('stardist_scale_')]
    
    if not scale_dirs:
        print("⏳ 还没有开始训练任何尺度")
        return True
    
    print(f"\n📈 各尺度训练状态:")
    
    for scale_dir in sorted(scale_dirs):
        scale_path = os.path.join(model_dir, scale_dir)
        scale = scale_dir.replace('stardist_scale_', '').replace('_', '.')
        
        print(f"\n🔍 尺度 {scale}:")
        
        # 检查训练日志
        log_path = os.path.join(scale_path, 'training_log.json')
        if os.path.exists(log_path):
            try:
                with open(log_path, 'r') as f:
                    log_data = json.load(f)
                
                start_time = log_data.get('start_time', '')
                current_time = log_data.get('current_time', '')
                epoch = log_data.get('epoch', 0)
                
                print(f"   ✅ 训练完成")
                print(f"   📅 开始时间: {start_time}")
                print(f"   📅 完成时间: {current_time}")
                print(f"   🔄 训练轮数: {epoch}")
                
                if 'history' in log_data and log_data['history']:
                    history = log_data['history']
                    if 'loss' in history:
                        final_loss = history['loss'][-1]
                        print(f"   📉 最终损失: {final_loss:.4f}")
                    if 'val_loss' in history:
                        final_val_loss = history['val_loss'][-1]
                        print(f"   📉 最终验证损失: {final_val_loss:.4f}")
                
            except Exception as e:
                print(f"   ⚠️ 读取训练日志失败: {e}")
        
        # 检查训练摘要
        summary_path = os.path.join(scale_path, 'training_summary.json')
        if os.path.exists(summary_path):
            try:
                with open(summary_path, 'r') as f:
                    summary = json.load(f)
                
                print(f"   ⏱️ 训练耗时: {summary.get('training_time', 0):.2f}秒")
                print(f"   🎯 检测对象数: {summary.get('n_detected_objects', 0)}")
                print(f"   📊 训练样本数: {summary.get('n_train_samples', 0)}")
                print(f"   📊 验证样本数: {summary.get('n_val_samples', 0)}")
                
            except Exception as e:
                print(f"   ⚠️ 读取训练摘要失败: {e}")
        
        # 检查权重文件
        weights_path = os.path.join(scale_path, 'weights_best.h5')
        if os.path.exists(weights_path):
            file_size = os.path.getsize(weights_path) / (1024 * 1024)  # MB
            mod_time = datetime.fromtimestamp(os.path.getmtime(weights_path))
            print(f"   💾 权重文件: {file_size:.2f}MB (更新于 {mod_time.strftime('%H:%M:%S')})")
        else:
            print(f"   ⏳ 权重文件不存在，可能还在训练中...")
    
    return True

def monitor_training(model_dir, interval=30):
    """持续监控训练"""
    print(f"开始监控训练进度...")
    print(f"模型目录: {model_dir}")
    print(f"检查间隔: {interval}秒")
    print(f"按 Ctrl+C 停止监控")
    
    try:
        while True:
            success = check_training_progress(model_dir)
            if not success:
                break
                
            print(f"\n⏰ 下次检查时间: {(datetime.now().timestamp() + interval)}")
            print("按 Ctrl+C 停止监控...")
            
            time.sleep(interval)
            
    except KeyboardInterrupt:
        print(f"\n👋 监控已停止")

def main():
    args = parse_args()
    
    if args.interval <= 0:
        # 单次检查
        check_training_progress(args.model_dir)
    else:
        # 持续监控
        monitor_training(args.model_dir, args.interval)

if __name__ == "__main__":
    main()

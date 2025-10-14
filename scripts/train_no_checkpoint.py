"""
禁用检查点保存的训练测试
"""
import argparse
import os
import numpy as np
from skimage import io
from csbdeep.utils import normalize_mi_ma
from stardist.models import StarDist2D, Config2D

# 修改SaveCallback来避免h5py问题
class NoSaveCallback:
    def on_epoch_end(self, epoch, logs=None):
        print(f"Epoch {epoch} completed, loss: {logs.get('loss', 'N/A')}")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--slides", type=str, required=True) 
    parser.add_argument("--n_epochs", type=int, default=3)
    parser.add_argument("--max_samples", type=int, default=6)
    return parser.parse_args()

def load_data(data_path, slides, max_samples=6):
    """加载数据"""
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
        
        from skimage.transform import resize
        img_small = resize(img, (128, 128), preserve_range=True).astype(np.float32)
        mask_small = resize(mask, (128, 128), preserve_range=True).astype(np.uint8)
        
        img_small = normalize_mi_ma(img_small, 0, 255)
        mask_small = (mask_small // 255).astype(np.uint8)
        
        X.append(img_small)
        Y.append(mask_small)
        
        print(f"  样本 {i+1}: {img_small.shape}")
    
    return np.array(X), np.array(Y)

def main():
    args = parse_args()
    
    print("=== 无检查点训练测试 ===")
    
    X, Y = load_data(args.data_path, args.slides, args.max_samples)
    print(f"数据: X {X.shape}, Y {Y.shape}")
    
    # 创建配置
    conf = Config2D(
        n_rays=16,
        grid=(1, 1),
        n_channel_in=3,
        train_patch_size=(64, 64),
        train_batch_size=2,
        train_learning_rate=0.001,
        train_epochs=args.n_epochs,
        train_steps_per_epoch=3
    )
    
    print(f"配置: epochs={args.n_epochs}, steps_per_epoch=3")
    
    # 创建模型但不保存
    temp_dir = "/tmp/no_save_test"
    os.makedirs(temp_dir, exist_ok=True)
    
    model = StarDist2D(conf, name='no_save', basedir=temp_dir)
    
    print("开始训练（禁用保存）...")
    
    try:
        # 手动训练循环，避免自动保存
        from stardist.models.model2d import StarDistData2D
        
        # 创建数据生成器
        data_gen = StarDistData2D(X, Y, batch_size=conf.train_batch_size, 
                                  n_rays=conf.n_rays, grid=conf.grid, 
                                  patch_size=conf.train_patch_size, length=len(X))
        
        # 编译模型
        model.keras_model.compile(
            optimizer='adam',
            loss=model._loss,
            metrics=model._metrics
        )
        
        # 手动训练几个步骤
        for epoch in range(args.n_epochs):
            print(f"\nEpoch {epoch+1}/{args.n_epochs}")
            
            epoch_losses = []
            for step in range(3):  # 只训练3步
                batch = next(iter(data_gen))
                X_batch, Y_batch = batch
                
                # 训练一步
                loss = model.keras_model.train_on_batch(X_batch, Y_batch)
                epoch_losses.append(loss)
                
                print(f"  Step {step+1}/3: loss = {loss}")
            
            avg_loss = np.mean(epoch_losses)
            print(f"  平均loss: {avg_loss:.4f}")
        
        print("✅ 训练成功完成！")
        
        # 测试预测
        print("测试预测...")
        labels, _ = model.predict_instances(X[0])
        print(f"预测成功: {labels.shape}, 实例数: {len(np.unique(labels))-1}")
        
        return True
        
    except Exception as e:
        print(f"❌ 训练失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)

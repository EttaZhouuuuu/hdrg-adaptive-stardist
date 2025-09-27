import argparse
import os
import sys
import numpy as np
from glob import glob
from skimage import io
from sklearn.model_selection import train_test_split
from skimage.transform import resize
from csbdeep.utils import Path, normalize_mi_ma

from stardist import fill_label_holes, random_label_cmap, calculate_extents
from stardist.models import Config2D, StarDist2D
from train_utils import augmenter, plot_img_label

np.random.seed(42)
lbl_cmap = random_label_cmap()

from constants import WANDB_KEY, WANDB_ENTITY, WANDB_PROJECT_NAME, CLASS_LABELS
from wandb_helper import WandbSegEvalCallback
import wandb
wandb.login(key=WANDB_KEY)

parser = argparse.ArgumentParser()
parser.add_argument("--exp_id", type=str, required=True)
parser.add_argument("--wandb_run_name", type=str, required=True)
parser.add_argument("--data_path", type=str, required=True)
parser.add_argument("--slides", type=str, nargs='+', required=True)
parser.add_argument("--ckpt_path", type=str, required=True)
parser.add_argument("--target_image_len", type=int, default=256)
parser.add_argument("--random_seed", type=int, default=42)
parser.add_argument("--n_rays", type=int, default=32)
parser.add_argument("--use_gpu", action='store_true')
parser.add_argument("--use_wandb", action='store_true')
# parser.add_argument("--grid_size", type=tuple, default=(4, 4))
parser.add_argument("--grid_size", nargs=2, type=int, default=[4, 4],
                    help="Grid size as two integers, e.g. --grid_size 4 4")
parser.add_argument("--train_batch_size", type=int, default=4)
parser.add_argument("--train_lr", type=float, default=3e-4)
parser.add_argument("--n_epochs", type=int, default=50)
parser.add_argument("--train_steps_per_epoch", type=int, default=100)
args = parser.parse_args()

EXP_ID              = args.exp_id
WANDB_RUN_NAME      = args.wandb_run_name
DATA_PATH           = args.data_path
SLIDES              = args.slides
CKPT_PATH           = args.ckpt_path
TARGET_IMAGE_SIZE   = (args.target_image_len, args.target_image_len)
RANDOM_SEED         = args.random_seed
N_RAYS              = args.n_rays
USE_GPU             = args.use_gpu
USE_WANDB           = args.use_wandb
GRID_SIZE           = args.grid_size
TRAIN_BATCH_SIZE    = args.train_batch_size
TRAIN_LR            = args.train_lr
N_EPOCHS            = args.n_epochs
TRAIN_STEPS_PER_EPOCH = args.train_steps_per_epoch


if not os.path.exists(CKPT_PATH):
    os.makedirs(CKPT_PATH)

IMAGE_PATHS = [os.path.join(DATA_PATH, image_name, 'patches_2048')
               for image_name in os.listdir(DATA_PATH)
               if image_name in SLIDES]

MASK_PATHS = [os.path.join(DATA_PATH, image_name, 'pseudo_gt_masks_2048')
              for image_name in os.listdir(DATA_PATH)
              if image_name in SLIDES]
X = []
for IMAGE_PATH in IMAGE_PATHS:
    X += glob(os.path.join(IMAGE_PATH, '*.png'))
X = sorted(X)

Y = []
for MASK_PATH in MASK_PATHS:
    Y += glob(os.path.join(MASK_PATH, '*.png'))
Y = sorted(Y)

assert all(Path(x).name==Path(y).name for x,y in zip(X,Y))
X = [io.imread(x) for x in X]
Y = [io.imread(y) for y in Y]
Y = [(mask // 255).astype(np.uint8) for mask in Y] # 0: background, 1: object
n_channel = 1 if X[0].ndim == 2 else X[0].shape[-1]

# normalize images (must be done before resizing)
X = [normalize_mi_ma(x, 0, 255) for x in X]
Y = [fill_label_holes(y) for y in Y]

# resize images and masks
X = [resize(x, TARGET_IMAGE_SIZE, anti_aliasing=False).astype(np.float32)
     for x in X]
Y = [(resize(y, TARGET_IMAGE_SIZE, anti_aliasing=False) > 0).astype(np.uint8)
     for y in Y]

X_trn, X_test, Y_trn, Y_test = train_test_split(X, Y, test_size=0.15,
                                                random_state=RANDOM_SEED)
print('number of images: %3d' % len(X))
print('- training:       %3d' % len(X_trn))
print('- validation:     %3d' % len(X_test))

# Display an example input (image, mask) pair
i = 0
img, lbl = X[i], Y[i]
assert img.ndim in (2,3)
img = img if (img.ndim==2 or img.shape[-1]==3) else img[...,0]
plot_img_label(img,lbl)
None;

conf = Config2D (
    n_rays       = N_RAYS,
    grid         = GRID_SIZE,
    use_gpu      = USE_GPU,
    n_channel_in = n_channel,
    train_patch_size = TARGET_IMAGE_SIZE,
    train_batch_size = TRAIN_BATCH_SIZE,
    train_learning_rate = TRAIN_LR,
    train_steps_per_epoch = TRAIN_STEPS_PER_EPOCH
)

vars(conf)

model = StarDist2D(conf, name='stardist', basedir=CKPT_PATH)

median_size = calculate_extents(list(Y), np.median)
fov = np.array(model._axes_tile_overlap('YX'))
print(f"median object size:      {median_size}")
print(f"network field of view :  {fov}")
if any(median_size > fov):
    print("WARNING: median object size larger than field of view of the neural network.")

data_kwargs = dict (
    n_rays           = conf.n_rays,
    grid             = conf.grid,
    patch_resize     = conf.train_patch_size,
    batch_size       = conf.train_batch_size,
    learning_rate    = conf.train_learning_rate,
    use_gpu          = conf.use_gpu
)

if USE_WANDB:
    run = wandb.init(
        entity = WANDB_ENTITY,
        project = WANDB_PROJECT_NAME,
        name = WANDB_RUN_NAME,
        config = data_kwargs
    )

    customized_image_logger = WandbSegEvalCallback(
        model=model,
        validation_data=(X_trn,Y_trn),
        data_table_columns=["idx", "ground_truth"],
        pred_table_columns=["epoch", "idx", "prediction", "iou_score"],
        labels=CLASS_LABELS,
        num_to_log=5
    )

model.train(
    X_trn, Y_trn, validation_data=(X_test,Y_test), augmenter=augmenter,
    epochs=N_EPOCHS, use_wandb=USE_WANDB, wandb_entity=WANDB_ENTITY,
    wandb_project_name=WANDB_PROJECT_NAME, wandb_run_name=WANDB_RUN_NAME,
    # wandb_image_logger=customized_image_logger
)

if USE_WANDB:
    run.finish()
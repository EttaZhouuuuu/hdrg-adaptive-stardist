import numpy as np
import wandb
from wandb.integration.keras import WandbEvalCallback
from sklearn.metrics import jaccard_score

from stardist.models import StarDistData2D

def wb_mask(bg_img, pred_mask, true_mask, labels):
    return wandb.Image(bg_img, masks={
        "prediction" : {"mask_data" : pred_mask, "class_labels" : labels},
        "ground truth" : {"mask_data" : true_mask, "class_labels" : labels}
    })


class WandbSegEvalCallback(WandbEvalCallback):
    def __init__(
        self, model, validation_data, data_table_columns, pred_table_columns,
        labels, num_to_log=5
    ):
        super().__init__(data_table_columns, pred_table_columns)

        self.inner_model = model
        self.imgs = validation_data[0][:num_to_log]
        self.masks = validation_data[1][:num_to_log]
        self.labels = labels

    def add_ground_truth(self, logs=None):
        for idx, (img, true_mask) in enumerate(zip(self.imgs, self.masks)):
            self.data_table.add_data(idx, wandb.Image(img, masks={
                "ground truth" : {"mask_data" : true_mask, "class_labels" : self.labels}
            }))

    def add_model_predictions(self, epoch, logs=None):
        for idx, (img, true_mask) in enumerate(zip(self.imgs, self.masks)):
            # print("img shape: ", img.shape)
            # print("true_mask shape: ", true_mask.shape)
            pred_mask = (self.inner_model.predict_instances(img)[0] > 0).astype(np.uint8)
            # print("pred_mask shape: ", pred_mask.shape)
            iou = jaccard_score(true_mask.flatten(), pred_mask.flatten(), average='binary')
            self.pred_table.add_data(
                epoch, idx, wb_mask(img, pred_mask, true_mask, self.labels), iou
            )

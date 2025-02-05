import os
from typing import List

from dataset_manager import BoundingBox, DatasetManager

import numpy as np

import torch

from ultralytics import YOLO


class ActiveLearningManager:
    def __init__(self, model_path: str = 'yolo11n.pt', device: str = 'auto'):

        if device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        self.device = torch.device(self.device)
        self.model_dir = 'models'
        path = os.path.join(self.model_dir, model_path)
        self.model = YOLO(path).to(self.device)

    def predict(self, image: np.array) -> List[BoundingBox]:
        """Predict bounding boxes on an image."""
        predictions = self.model(image)[0]
        boxes = []
        YOLO_boxes = predictions.boxes
        for box in YOLO_boxes:
            x_min, y_min, x_max, y_max = box.xyxy.detach().tolist()[0]
            class_id = int(box.cls.detach().item())
            bounding_box = BoundingBox(x_min, y_min, x_max, y_max, class_id)
            boxes.append(bounding_box)
        return boxes

    def train(self, dataset_manager: DatasetManager):
        """Train the model with the dataset."""
        data_path = dataset_manager.data_yaml_path
        image_train_dir = dataset_manager.dataset_name + '/images/train'

        # make sure there is at least one file inside image_train_dir
        if len(os.listdir(image_train_dir)) <= 1:
            print('No images in the train folder')
            return

        learning_rate = 0.001
        batch_size = 8
        epochs = 10
        self.model.train(data=data_path,
                         epochs=epochs,
                         imgsz=640,
                         lr0=learning_rate,
                         device=self.device,
                         batch=batch_size,
                         cache=False,
                         degrees=0,
                         )

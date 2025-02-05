import os

from typing import List

from dataset_manager import BoundingBox

import numpy as np

from ultralytics import YOLO


class ActiveLearningManager:
    def __init__(self, model_path: str = 'yolo11n.pt'):

        self.model_dir = "models"
        path = os.path.join(self.model_dir, model_path)
        self.model = YOLO(path)

    def predict(self, image: np.array) -> List[BoundingBox]:
        predictions = self.model(image)[0]
        boxes = []
        YOLO_boxes = predictions.boxes
        for box in YOLO_boxes:
            x_min, y_min, x_max, y_max = box.xyxy.detach().tolist()[0]
            class_id = int(box.cls.detach().item())
            bounding_box = BoundingBox(x_min, y_min, x_max, y_max, class_id)
            boxes.append(bounding_box)
        return boxes

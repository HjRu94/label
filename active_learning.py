from typing import List

from dataset_manager import BoundingBox

import numpy as np

from ultralytics import YOLO


class ActiveLearningManager:
    def __init__(self, model_path: str = 'yolo11n.pt'):

        self.model = YOLO(model_path)

    def predict(self, image: np.array) -> List[BoundingBox]:
        predictions = self.model(image)[0]
        boxes = []
        YOLO_boxes = predictions.boxes
        for box in YOLO_boxes:
            x_min, y_min, x_max, y_max = box.xyxy.detach().tolist()[0]
            class_id = box.cls
            bounding_box = BoundingBox(x_min, y_min, x_max, y_max, class_id)
            boxes.append(bounding_box)
        return boxes

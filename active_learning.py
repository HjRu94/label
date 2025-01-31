import os

import cv2

import numpy as np


class BoundingBox:
    """Class to manage bounding boxes."""

    def __init__(self, x_min: int, y_min: int, x_max: int, y_max: int, class_id: int):
        self.x_min = x_min
        self.y_min = y_min
        self.x_max = x_max
        self.y_max = y_max
        self.class_id = class_id

    def to_yolo_format(self, image_width: int, image_height: int) -> tuple:
        """Convert bounding box to YOLO format."""
        x_center = (self.x_min + self.x_max) / 2
        y_center = (self.y_min + self.y_max) / 2
        width = self.x_max - self.x_min
        height = self.y_max - self.y_min

        x_center /= image_width
        y_center /= image_height
        width /= image_width
        height /= image_height

        return x_center, y_center, width, height, self.class_id

class ClassDescription:
    """Class to manage class names."""

    class_names = []
    n_classes = 0

    def __init__(self, class_names):
        self.set_class_names(class_names)

    def set_class_names(self, class_names):
        self.class_names = class_names
        self.n_classes = len(class_names)
        return self.n_classes


class DatasetManager:
    """Class to manage the dataset."""

    def __init__(self, dataset_name: str, class_description: ClassDescription):
        self.dataset_name = dataset_name
        self.class_description = class_description

    def create_folder_structure(self, include_train=True, include_validation=True, include_test=False):
        """
        Create the folder structure for the dataset.

        File structure looks like this:
        dataset_name
        ├── images
        │   ├── train
        │   ├── validation
        │   └── test
        ├── labels
        │   ├── train
        │   ├── validation
        │   └── test
        └── data.yaml
        """
        os.makedirs(self.dataset_name, exist_ok=True)
        if include_train:
            os.makedirs(os.path.join(self.dataset_name, 'images', 'train'), exist_ok=True)
            os.makedirs(os.path.join(self.dataset_name, 'labels', 'train'), exist_ok=True)
        if include_validation:
            os.makedirs(os.path.join(self.dataset_name, 'images', 'validation'), exist_ok=True)
            os.makedirs(os.path.join(self.dataset_name, 'labels', 'validation'), exist_ok=True)
        if include_test:
            os.makedirs(os.path.join(self.dataset_name, 'images', 'test'), exist_ok=True)
            os.makedirs(os.path.join(self.dataset_name, 'labels', 'test'), exist_ok=True)

        with open(os.path.join(self.dataset_name, 'data.yaml'), 'w') as f:
            f.write(f'path: {self.dataset_name}\n')
            if include_train:
                f.write('train: images/train\n')
            if include_validation:
                f.write('val: images/validation\n')
            if include_test:
                f.write('test: images/test\n')
            if self.class_description.n_classes > 0:
                f.write(f'nc: {self.class_description.n_classes}\n')
                f.write('names: ')
                f.write(str(self.class_description.class_names))
                f.write('\n')


class ImageManager:
    """Class to manage unlabelled images."""

    def __init__(self, folder_path: str):
        self.allowed_extensions = ['.jpg', '.jpeg', '.png']
        self.folder_path = folder_path
        self.image_paths = self.load_images(folder_path)
        self.current_image_index = 0

        self.cache = ('', None)

    def load_images(self, folder_path: str):
        """Load all images from the image folder recursively."""
        image_paths = []
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                if os.path.splitext(file)[1].lower() in self.allowed_extensions:
                    image_paths.append(os.path.join(root, file))
        return image_paths

    def load_image(self) -> np.ndarray:
        """Load the current image from file."""
        image_path = self.image_paths[self.current_image_index]
        if image_path == self.cache[0]:
            return self.cache[1]
        image = cv2.imread(image_path)
        return image


if __name__ == '__main__':
    class_description = ClassDescription(['cat', 'dog'])

    dataset_manager = DatasetManager('dataset', class_description)
    dataset_manager.create_folder_structure()

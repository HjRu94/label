import os
from typing import List

import cv2

import numpy as np


class BoundingBox:
    """Class to manage bounding boxes."""

    def __init__(self, x_min: float, y_min: float, x_max: float, y_max: float, class_id: int):
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

    def area(self):
        """Calculate the area of the bounding box."""
        return (self.x_max - self.x_min) * (self.y_max - self.y_min)

    def __str__(self):
        """Return the bounding box in a string format."""
        return f'{self.x_min} {self.y_min} {self.x_max} {self.y_max} {self.class_id}'


class ClassDescription:
    """Class to manage class names."""

    class_names = []
    n_classes = 0

    def __init__(self, class_names):
        self.set_class_names(class_names)

    def set_class_names(self, class_names):
        """Set the class names."""
        self.class_names = class_names
        self.n_classes = len(class_names)
        return self.n_classes


class DatasetManager:
    """Class to manage the dataset."""

    def __init__(self,
                 dataset_name: str,
                 class_description: ClassDescription,
                 include_train=True,
                 include_validation=True,
                 include_test=False,
                 train_split=0.6,
                 validation_split=0.2,
                 test_split=0.2
                 ) -> None:

        self.dataset_name = dataset_name
        self.class_description = class_description

        self.include_train = include_train
        self.include_validation = include_validation
        self.include_test = include_test

        self.train_split = 0.6 if include_train else 0
        self.validation_split = 0.2 if include_validation else 0
        self.test_split = 0.2 if include_test else 0

        self.image_number = 0

        self.last_image_paths = []
        self.last_lable_paths = []

    def create_folder_structure(self):
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
        if self.include_train:
            os.makedirs(os.path.join(self.dataset_name, 'images', 'train'), exist_ok=True)
            os.makedirs(os.path.join(self.dataset_name, 'labels', 'train'), exist_ok=True)
        if self.include_validation:
            os.makedirs(os.path.join(self.dataset_name, 'images', 'validation'), exist_ok=True)
            os.makedirs(os.path.join(self.dataset_name, 'labels', 'validation'), exist_ok=True)
        if self.include_test:
            os.makedirs(os.path.join(self.dataset_name, 'images', 'test'), exist_ok=True)
            os.makedirs(os.path.join(self.dataset_name, 'labels', 'test'), exist_ok=True)

        with open(os.path.join(self.dataset_name, 'data.yaml'), 'w') as f:
            f.write(f'path: {self.dataset_name}\n')
            if self.include_train:
                f.write('train: images/train\n')
            if self.include_validation:
                f.write('val: images/validation\n')
            if self.include_test:
                f.write('test: images/test\n')
            if self.class_description.n_classes > 0:
                f.write(f'nc: {self.class_description.n_classes}\n')
                f.write('names: ')
                f.write(str(self.class_description.class_names))
                f.write('\n')

    def save_image(self, image: np.ndarray, bounding_boxes: List[BoundingBox]):
        """Save the image to file."""
        # Deside where to save the image with a random number
        random_number = np.random.rand()
        random_number *= self.train_split + self.validation_split + self.test_split

        if random_number < self.train_split:
            folder = 'train'
        elif random_number < self.train_split + self.validation_split:
            folder = 'validation'
        else:
            folder = 'test'

        image_path = os.path.join(self.dataset_name, 'images', folder, f'image{self.image_number}.jpg')
        # Save the image
        cv2.imwrite(image_path, image)

        # Save the bounding boxes under labels
        label_path = os.path.join(self.dataset_name, 'labels', folder, f'image{self.image_number}.txt')

        self.last_image_paths.append(image_path)
        self.last_lable_paths.append(label_path)
        with open(label_path, 'w') as f:
            for bounding_box in bounding_boxes:
                x_center, y_center, width, height, class_id = bounding_box.to_yolo_format(image.shape[1], image.shape[0])
                f.write(f'{class_id} {x_center} {y_center} {width} {height}\n')
        self.image_number += 1

    def remove_last_image(self):
        """Remove the last image from the dataset."""
        if len(self.last_image_paths) == 0 or len(self.last_lable_paths) == 0:
            print('No images to remove')
            return
        self.image_number -= 1
        image_path = self.last_image_paths.pop()
        label_path = self.last_lable_paths.pop()
        os.remove(image_path)
        os.remove(label_path)


class ImageManager:
    """Class to manage unlabelled images."""

    def __init__(self, folder_path: str):
        self.allowed_extensions = ['.jpg', '.jpeg', '.png']
        self.folder_path = folder_path
        self.image_paths = self.load_images(folder_path)
        self.current_image_index = 0

        self.cache = ('', None)
        self.removed_images = []

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
        self.cache = (image_path, image)
        return image

    def next_image(self):
        """Load the next image."""
        removed_path, removed_image = self.image_paths[self.current_image_index], self.load_image()
        self.removed_images.append((removed_path, removed_image))
        os.remove(removed_path)
        self.image_paths.pop(self.current_image_index)
        self.cache = ('', None)

    def previous_image(self):
        """Load the previous image."""
        if len(self.removed_images) == 0:
            print('No images to recover')
            return
        recovered_path, recovered_image = self.removed_images.pop()
        # Save the recovered image
        cv2.imwrite(recovered_path, recovered_image)

        # Insert the recovered image back to the list
        self.image_paths.insert(self.current_image_index, recovered_path)


if __name__ == '__main__':
    class_description = ClassDescription(['cat', 'dog'])

    dataset_manager = DatasetManager('dataset', class_description)
    dataset_manager.create_folder_structure()

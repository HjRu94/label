import os


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


if __name__ == '__main__':
    class_description = ClassDescription(['cat', 'dog'])

    dataset_manager = DatasetManager('dataset', class_description)
    dataset_manager.create_folder_structure()

from dataset_manager import ClassDescription, DatasetManager, ImageManager

import gui


def label_images():
    image_dir = 'images'
    dataset_dir = 'dataset'
    classes = ['cat', 'dog', 'bird', 'fish', 'rabbit', 'hamster', 'turtle', 'horse', 'cow', 'pig',
               'elephant', 'giraffe', 'zebra', 'bear', 'lion', 'tiger', 'wolf', 'fox', 'deer', 'monkey']
    screen_size = (800, 600)
    default_scale = 2

    class_descrition = ClassDescription(classes)
    dataset_manager = DatasetManager(dataset_dir, class_descrition)

    dataset_manager.create_folder_structure()
    image_manager = ImageManager(image_dir)

    my_gui = gui.ImageLabeler(image_manager, dataset_manager, screen_size, default_scale)
    my_gui.loop()


if __name__ == '__main__':
    label_images()

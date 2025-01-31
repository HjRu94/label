from active_learning import ClassDescription, DatasetManager, ImageManager

import gui


def label_images():
    image_dir = 'images'
    dataset_dir = 'dataset'
    classes = ['cat', 'dog', 'bird']

    class_descrition = ClassDescription(classes)
    dataset_manager = DatasetManager(dataset_dir, class_descrition)

    dataset_manager.create_folder_structure()
    image_manager = ImageManager(image_dir)

    my_gui = gui.ImageLabeler(image_manager, dataset_manager)
    my_gui.loop()


if __name__ == '__main__':
    label_images()

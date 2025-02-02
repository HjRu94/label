from dataset_manager import ClassDescription, DatasetManager, ImageManager
from active_learning import ActiveLearningManager

import gui


def label_images(args):
    image_dir = args.image_dir
    dataset_dir = args.dataset_dir
    classes = args.classes
    screen_size = args.screen_size
    default_scale = args.default_scale

    class_descrition = ClassDescription(classes)
    dataset_manager = DatasetManager(dataset_dir, class_descrition)
    active_learning_manager = ActiveLearningManager()

    dataset_manager.create_folder_structure()
    image_manager = ImageManager(image_dir)

    my_gui = gui.ImageLabeler(image_manager, dataset_manager,active_learning_manager, screen_size, default_scale)
    my_gui.loop()


if __name__ == '__main__':
    label_images()

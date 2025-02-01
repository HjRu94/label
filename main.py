import argparse

from label import label_images


def main():
    """script to parse command line arguments"""
    parser = argparse.ArgumentParser(description='Choose options')

    subparsers = parser.add_subparsers(dest='command')

    label_parser = subparsers.add_parser('label', help='Label images')
    label_parser.add_argument('--image_dir', default='images', help='Directory with images')
    label_parser.add_argument('--dataset_dir', default='dataset', help='Directory with dataset')
    label_parser.add_argument('--classes', default=['class0', 'class1'], type=str, nargs='+', help='Classes')
    label_parser.add_argument('--screen_size', default=[1200, 800], type=int, nargs=2, help='Screen size')
    label_parser.add_argument('--default_scale', default=1.0, type=float, help='Default scale')
    # add function
    label_parser.set_defaults(func=label_images)

    args = parser.parse_args()
    if args.command is None:
        parser.print_help()
    else:
        args.func(args)


if __name__ == '__main__':
    main()

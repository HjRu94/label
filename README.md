# Object Detection Labeling Tool

A Python-based labeling tool designed for creating bounding box annotations for object detection tasks.

## Features

- Click-and-drag interface for drawing bounding boxes.
- Easy class management with custom class options.
- Intuitive controls for modifying and navigating annotations.

## Installation

1. Place all your image files into a folder named `images`.
2. Set up a virtual environment:
   ```bash
   python3 -m venv myenv
   source myenv/bin/activate
   pip install -r requirements.txt
   ```

## Usage

### Basic Labeling Command

Run the following command to start labeling:

```bash
python main.py label
```

### Adding Custom Classes

Specify your custom classes using the `--classes` flag:

```bash
python main.py label --classes dog cat bird
```

## Labeling Controls

- **Draw Bounding Box**: Click and drag with the left mouse button.
- **Remove Bounding Box**: Hold `Shift` and left-click on the bounding box.
- **Pan/Move Around**: Click and drag with the left mouse button.
- **Zoom In/Out**: Scroll with the mouse wheel.

## License

This project is licensed under the MIT License.


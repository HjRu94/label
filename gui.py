from typing import List, Tuple

from active_learning import ImageManager, BoundingBox, DatasetManager

import pygame as pg


class ImageLabeler:
    def __init__(self, image_manager: ImageManager, dataset_manager: DatasetManager):
        pg.init()
        self.screen_size = (800, 600)
        self.screen = pg.display.set_mode(self.screen_size)
        self.running = True
        self.image_manager = image_manager
        self.dataset_manager = dataset_manager

        self.bounding_boxes: List[BoundingBox] = []  # List to store bounding boxes
        self.drawing = False  # Track if we are drawing a box
        self.start_pos = None  # Start position of the box
        self.current_box = None  # Current box being drawn

        # Transformation parameters
        self.scale = 1.0
        self.offset = [0, 0]  # x, y offset
        self.initialize_image_pos()
        self.moving = False
        self.last_mouse_pos = None

    def apply_transform(self, pos: Tuple[int, int]) -> Tuple[float, float]:
        """Apply scale and offset to a position."""
        x, y = pos
        return (x - self.offset[0]) / self.scale, (y - self.offset[1]) / self.scale

    def initialize_image_pos(self):
        self.scale = 1.0
        image = self.image_manager.load_image()
        self.offset = [self.screen_size[0] / 2 - image.shape[1] / 2, self.screen_size[1] / 2 - image.shape[0] / 2]

    def event(self):
        """Handle processing events."""
        for event in pg.event.get():
            if event.type == pg.QUIT:
                self.running = False
            elif event.type == pg.MOUSEBUTTONDOWN:
                if event.button == 1:  # Left mouse button for drawing
                    self.drawing = True
                    self.start_pos = self.apply_transform(event.pos)
                    self.current_box = None
                elif event.button == 3:  # Right mouse button for moving the image
                    self.moving = True
                    self.last_mouse_pos = event.pos
            elif event.type == pg.MOUSEBUTTONUP:
                if event.button == 1 and self.start_pos:
                    end_pos = self.apply_transform(event.pos)
                    x1, y1 = self.start_pos
                    x2, y2 = end_pos
                    x, y = min(x1, x2), min(y1, y2)
                    width, height = abs(x2 - x1), abs(y2 - y1)
                    self.bounding_boxes.append(BoundingBox(x, y, x + width, y + height, 0))
                    self.drawing = False
                    self.start_pos = None
                    self.current_box = None
                elif event.button == 3:  # Stop moving the image
                    self.moving = False
            elif event.type == pg.MOUSEMOTION:
                if self.drawing and self.start_pos:
                    end_pos = self.apply_transform(event.pos)
                    x1, y1 = self.start_pos
                    x2, y2 = end_pos
                    x, y = min(x1, x2), min(y1, y2)
                    width, height = abs(x2 - x1), abs(y2 - y1)
                    self.current_box = (x, y, width, height)
                if self.moving and self.last_mouse_pos:
                    dx, dy = event.pos[0] - self.last_mouse_pos[0], event.pos[1] - self.last_mouse_pos[1]
                    self.offset[0] += dx
                    self.offset[1] += dy
                    self.last_mouse_pos = event.pos

            if event.type == pg.MOUSEWHEEL:
                mouse_x, mouse_y = pg.mouse.get_pos()
                pre_zoom_mouse_pos = self.apply_transform((mouse_x, mouse_y))

                if event.y > 0:  # Scroll up to zoom in
                    scale_factor = 1.1
                else:  # Scroll down to zoom out
                    scale_factor = 0.9

                self.scale *= scale_factor

                # Adjust offset to keep the image under the mouse stationary
                self.offset[0] = mouse_x - pre_zoom_mouse_pos[0] * self.scale
                self.offset[1] = mouse_y - pre_zoom_mouse_pos[1] * self.scale

            if event.type == pg.KEYDOWN:
                if event.key == pg.K_s:
                    image = self.image_manager.load_image()
                    self.dataset_manager.save_image(image, self.bounding_boxes)
                    self.image_manager.next_image(remove=False)
                    self.bounding_boxes = []
                    self.initialize_image_pos()

    def update(self):
        """Handle updating the screen."""
        pass

    def draw(self):
        """Handle drawing on the screen."""
        self.screen.fill((255, 255, 255))  # Clear screen

        # Draw current image
        image = self.image_manager.load_image()
        image = image.transpose((1, 0, 2))
        image = pg.surfarray.make_surface(image)

        # Apply scaling
        image = pg.transform.scale(image, (int(image.get_width() * self.scale), int(image.get_height() * self.scale)))

        # Apply offset
        self.screen.blit(image, self.offset)

        # Draw saved bounding boxes
        for box in self.bounding_boxes:
            rect = (
                int(box.x_min * self.scale + self.offset[0]),
                int(box.y_min * self.scale + self.offset[1]),
                int((box.x_max - box.x_min) * self.scale),
                int((box.y_max - box.y_min) * self.scale),
            )
            pg.draw.rect(self.screen, (255, 0, 0), rect, 2)

        # Draw the current box if in progress
        if self.current_box:
            rect = (
                int(self.current_box[0] * self.scale + self.offset[0]),
                int(self.current_box[1] * self.scale + self.offset[1]),
                int(self.current_box[2] * self.scale),
                int(self.current_box[3] * self.scale),
            )
            pg.draw.rect(self.screen, (0, 255, 0), rect, 2)

        pg.display.flip()

    def loop(self):
        """Main loop of the GUI."""
        while self.running:
            self.event()
            self.update()
            self.draw()
            pg.display.update()
        pg.quit()


if __name__ == "__main__":
    image_manager = ImageManager()  # Replace with actual initialization
    dataset_manager = DatasetManager()  # Replace with actual initialization
    my_gui = ImageLabeler(image_manager, dataset_manager)
    my_gui.loop()

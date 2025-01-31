from typing import List

from active_learning import ImageManager, BoundingBox

import pygame as pg


class ImageLabeler():
    def __init__(self, image_manager: ImageManager):
        pg.init()
        self.screen_size = (800, 600)
        self.screen = pg.display.set_mode(self.screen_size)
        self.running = True
        self.image_manager = image_manager

        self.bounding_boxes: List[BoundingBox] = []  # List to store bounding boxes
        self.drawing = False  # Track if we are drawing a box
        self.start_pos = None  # Start position of the box
        self.current_box = None  # Current box being drawn

    def event(self):
        """Handle processing events."""
        for event in pg.event.get():
            if event.type == pg.QUIT:
                self.running = False
            elif event.type == pg.MOUSEBUTTONDOWN:
                if event.button == 1:  # Left mouse button
                    self.drawing = True
                    self.start_pos = event.pos
                    self.current_box = None
            elif event.type == pg.MOUSEBUTTONUP:
                if event.button == 1 and self.start_pos:
                    end_pos = event.pos
                    bounxing_box = BoundingBox(self.start_pos[0], self.start_pos[1], end_pos[0], end_pos[1], 0)
                    self.bounding_boxes.append(bounxing_box)
                    self.drawing = False
                    self.start_pos = None
                    self.current_box = None
            elif event.type == pg.MOUSEMOTION and self.drawing:
                end_pos = event.pos
                self.current_box = (*self.start_pos, end_pos[0] - self.start_pos[0], end_pos[1] - self.start_pos[1])

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
        self.screen.blit(image, (0, 0))

        # Draw saved bounding boxes
        for box in self.bounding_boxes:
            box = pg.Rect(box.x_min, box.y_min, box.x_max - box.x_min, box.y_max - box.y_min)
            pg.draw.rect(self.screen, (255, 0, 0), box, 2)

        # Draw the current box if in progress
        if self.current_box:
            pg.draw.rect(self.screen, (0, 255, 0), self.current_box, 2)

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
    my_gui = ImageLabeler()
    my_gui.loop()

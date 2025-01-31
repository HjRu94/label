from active_learning import ImageManager

import pygame as pg


class ImageLabeler():
    def __init__(self, image_manager: ImageManager):
        pg.init()
        self.screen_size = (800, 600)
        self.screen = pg.display.set_mode(self.screen_size)
        self.running = True

        self.image_manager = image_manager

    def event(self):
        """Handle processing events."""
        for event in pg.event.get():
            if event.type == pg.QUIT:
                self.running = False

    def update(self):
        """Handle updating the screen."""
        pass

    def draw(self):
        """Handle drawing on the screen."""
        # Fill the screen with white color
        self.screen.fill((255, 255, 255))

        pg.display.flip()

    def loop(self):
        """Main loop of the gui."""
        while self.running:
            self.event()
            self.update()
            self.draw()
            pg.display.update()
        pg.quit()


if __name__ == "__main__":
    my_gui = ImageLabeler()
    my_gui.loop()

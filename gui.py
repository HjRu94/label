import pygame as pg


class ImageLabeler():
    def __init__(self):
        pg.init()
        self.screen_size = (800, 600)
        self.screen = pg.display.set_mode(self.screen_size)
        self.running = True

    def event(self):
        """Handle processing events."""
        pass

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
            pg.clock.tick(60)
        pg.quit()


if __name__ == "__main__":
    my_gui = ImageLabeler()
    my_gui.loop()

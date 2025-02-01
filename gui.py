from typing import List, Tuple

from active_learning import ImageManager, BoundingBox, DatasetManager

import pygame as pg


class ImageLabeler:
    def __init__(self, image_manager: ImageManager, dataset_manager: DatasetManager):
        pg.init()
        self.menu_height = 200
        self.menu_width = 300
        self.screen_size = (800 + self.menu_width, 800)
        self.screen = pg.display.set_mode(self.screen_size)
        self.running = True
        self.image_manager = image_manager
        self.dataset_manager = dataset_manager

        self.bounding_boxes: List[BoundingBox] = []
        self.drawing = False
        self.start_pos = None
        self.current_box = None

        self.scale = 1.0
        self.offset = [0, 0]
        self.initialize_image_pos()
        self.moving = False
        self.last_mouse_pos = None

        # Button setup
        self.buttons = [f"Animal {i+1}" for i in range(20)]
        self.current_class = 0
        self.class_counts = [0 for _ in range(20)]
        self.menu_scroll_offset = 0

    def apply_transform(self, pos: Tuple[int, int]) -> Tuple[float, float]:
        x, y = pos
        return (x - self.offset[0]) / self.scale, (y - self.offset[1]) / self.scale

    def initialize_image_pos(self):
        self.scale = 1.0
        image = self.image_manager.load_image()
        self.offset = [
            (self.screen_size[0] - self.menu_width) / 2 - image.shape[1] / 2,
            (self.screen_size[1] - self.menu_height) / 2 - image.shape[0] / 2,
        ]

    def event(self):
        self.mouse_pos = pg.mouse.get_pos()

        for event in pg.event.get():
            if event.type == pg.QUIT:
                self.running = False

            if self.mouse_pos[0] > self.screen_size[0] - self.menu_width:
                if event.type == pg.MOUSEBUTTONDOWN and event.button == 1:
                    self.check_button_click(self.mouse_pos)
                    self.check_navigation_click(self.mouse_pos)
                if event.type == pg.MOUSEWHEEL:
                    self.menu_scroll_offset -= event.y * 20
                    self.menu_scroll_offset = max(0, self.menu_scroll_offset)
                continue

            if event.type == pg.MOUSEBUTTONDOWN:
                if event.button == 1 and self.is_mouse_on_image(event.pos):
                    self.drawing = True
                    self.start_pos = self.apply_transform(event.pos)
                    self.current_box = None
                elif event.button == 3:
                    self.moving = True
                    self.last_mouse_pos = event.pos

            elif event.type == pg.MOUSEBUTTONUP:
                if event.button == 1 and self.start_pos:
                    end_pos = self.apply_transform(event.pos)
                    end_pos = self.clamp_to_image_bounds(end_pos)
                    x1, y1 = self.start_pos
                    x2, y2 = end_pos
                    x, y = min(x1, x2), min(y1, y2)
                    width, height = abs(x2 - x1), abs(y2 - y1)
                    self.bounding_boxes.append(BoundingBox(x, y, x + width, y + height, self.current_class))
                    self.class_counts[self.current_class] += 1
                    self.drawing = False
                    self.start_pos = None
                    self.current_box = None
                elif event.button == 3:
                    self.moving = False
                    self.last_mouse_pos = None

            elif event.type == pg.MOUSEMOTION:
                if self.drawing and self.start_pos:
                    end_pos = self.apply_transform(event.pos)
                    end_pos = self.clamp_to_image_bounds(end_pos)
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
                if self.mouse_pos[0] <= self.screen_size[0] - self.menu_width:
                    # Scale image when mouse is not over the menu
                    scale_factor = 1.1 if event.y > 0 else 0.9
                    mouse_x, mouse_y = self.mouse_pos
                    pre_zoom_mouse_pos = self.apply_transform((mouse_x, mouse_y))

                    self.scale *= scale_factor

                    self.offset[0] = mouse_x - pre_zoom_mouse_pos[0] * self.scale
                    self.offset[1] = mouse_y - pre_zoom_mouse_pos[1] * self.scale

    def check_button_click(self, pos):
        button_height = 50
        for i, label in enumerate(self.buttons):
            y_offset = 10 + i * (button_height + 10) - self.menu_scroll_offset
            button_rect = pg.Rect(
                self.screen_size[0] - self.menu_width + 10,
                y_offset,
                self.menu_width - 20,
                button_height
            )
            if button_rect.collidepoint(pos):
                self.current_class = i

    def check_navigation_click(self, pos):
        button_width = 100
        button_height = 50
        prev_button_rect = pg.Rect(self.screen_size[0] - self.menu_width + 10, self.screen_size[1] - button_height - 10, button_width, button_height)
        next_button_rect = pg.Rect(self.screen_size[0] - button_width - 10, self.screen_size[1] - button_height - 10, button_width, button_height)

        if prev_button_rect.collidepoint(pos):
            print("Previous button clicked")
        elif next_button_rect.collidepoint(pos):
            print("Next button clicked")

    def is_mouse_on_image(self, pos: Tuple[int, int]) -> bool:
        image = self.image_manager.load_image()
        return (
            self.offset[0] <= pos[0] <= self.offset[0] + image.shape[1] * self.scale
            and self.offset[1] <= pos[1] <= self.offset[1] + image.shape[0] * self.scale
        )

    def clamp_to_image_bounds(self, pos: Tuple[float, float]) -> Tuple[float, float]:
        image = self.image_manager.load_image()
        return (
            max(0, min(image.shape[1], pos[0])),
            max(0, min(image.shape[0], pos[1]))
        )

    def update(self):
        pass

    def draw(self):
        self.screen.fill((255, 255, 255))

        image = self.image_manager.load_image()
        visible_surface = pg.surfarray.make_surface(image.transpose((1, 0, 2)))
        visible_surface = pg.transform.scale(
            visible_surface,
            (int(image.shape[1] * self.scale), int(image.shape[0] * self.scale))
        )
        self.screen.blit(visible_surface, self.offset)

        for box in self.bounding_boxes:
            rect = (
                int(box.x_min * self.scale + self.offset[0]),
                int(box.y_min * self.scale + self.offset[1]),
                int((box.x_max - box.x_min) * self.scale),
                int((box.y_max - box.y_min) * self.scale),
            )
            pg.draw.rect(self.screen, (255, 0, 0), rect, 2)

        if self.current_box:
            rect = (
                int(self.current_box[0] * self.scale + self.offset[0]),
                int(self.current_box[1] * self.scale + self.offset[1]),
                int(self.current_box[2] * self.scale),
                int(self.current_box[3] * self.scale),
            )
            pg.draw.rect(self.screen, (0, 255, 0), rect, 2)

        pg.draw.rect(
            self.screen, (180, 180, 180),
            (self.screen_size[0] - self.menu_width, 0, self.menu_width, self.screen_size[1])
        )

        button_height = 50
        for i, label in enumerate(self.buttons):
            y_offset = 10 + i * (button_height + 10) - self.menu_scroll_offset
            button_rect = pg.Rect(
                self.screen_size[0] - self.menu_width + 10,
                y_offset,
                self.menu_width - 20,
                button_height
            )
            if button_rect.bottom < 0 or button_rect.top > self.screen_size[1] - 70:
                continue
            color = (100, 200, 100) if self.current_class == i else (150, 150, 150)
            pg.draw.rect(self.screen, color, button_rect)

            font = pg.font.SysFont(None, 24)
            text_surface = font.render(f"{label} ({self.class_counts[i]})", True, (0, 0, 0))
            text_rect = text_surface.get_rect(center=button_rect.center)
            self.screen.blit(text_surface, text_rect)

        button_width = 100
        button_height = 50
        prev_button_rect = pg.Rect(self.screen_size[0] - self.menu_width + 10, self.screen_size[1] - button_height - 10, button_width, button_height)
        next_button_rect = pg.Rect(self.screen_size[0] - button_width - 10, self.screen_size[1] - button_height - 10, button_width, button_height)

        pg.draw.rect(self.screen, (150, 150, 250), prev_button_rect)
        pg.draw.rect(self.screen, (150, 150, 250), next_button_rect)

        prev_text = pg.font.SysFont(None, 24).render("Previous", True, (0, 0, 0))
        next_text = pg.font.SysFont(None, 24).render("Next", True, (0, 0, 0))

        self.screen.blit(prev_text, prev_text.get_rect(center=prev_button_rect.center))
        self.screen.blit(next_text, next_text.get_rect(center=next_button_rect.center))

        pg.display.flip()

    def loop(self):
        while self.running:
            self.event()
            self.update()
            self.draw()
        pg.quit()

from typing import List, Tuple

from dataset_manager import BoundingBox, DatasetManager, ImageManager

import numpy as np

import pygame as pg


class ColorManager:
    """Class to manage colors."""

    def __init__(self):
        """Initialize colors."""
        TURQUOISE = (0x40, 0xE0, 0xD0)
        SOFT_CORAL = (0xFF, 0x6F, 0x61)
        VIVID_SKY_BLUE = (0x00, 0xBF, 0xFF)
        PLUM = (0xD4, 0x70, 0xA2)
        APRICOT = (0xFB, 0xAE, 0x7E)
        MEDIUM_AQUAMARINE = (0x66, 0xCD, 0xAA)
        SALMON_PINK = (0xFF, 0x91, 0xA4)
        SPRING_GREEN = (0x00, 0xFA, 0x9A)
        LIGHT_GOLDENROD = (0xFA, 0xD6, 0x60)
        COOL_LAVENDER = (0xB5, 0x9D, 0xDD)
        SOFT_TANGERINE = (0xFF, 0xA0, 0x4C)
        BRIGHT_PERIWINKLE = (0x8A, 0x9A, 0xE6)
        TEAL_BLUE = (0x36, 0x9E, 0xB7)
        CRISP_ROSE = (0xFF, 0x75, 0x8A)
        MELON = (0xF7, 0x85, 0x6B)
        MODERATE_LIME_GREEN = (0x9A, 0xCD, 0x32)
        FRESH_CYAN = (0x00, 0xCF, 0xD0)
        SOFT_AMBER = (0xFF, 0xC1, 0x07)
        LILAC = (0xC8, 0xA2, 0xC8)
        VIBRANT_MINT = (0x3E, 0xD8, 0x92)
        self.colors = [TURQUOISE, SOFT_CORAL, VIVID_SKY_BLUE, PLUM, APRICOT, MEDIUM_AQUAMARINE, SALMON_PINK,
                       SPRING_GREEN, LIGHT_GOLDENROD, COOL_LAVENDER, SOFT_TANGERINE, BRIGHT_PERIWINKLE, TEAL_BLUE,
                       CRISP_ROSE, MELON, MODERATE_LIME_GREEN, FRESH_CYAN, SOFT_AMBER, LILAC, VIBRANT_MINT]
        self.index_list = list(range(len(self.colors)))
        np.random.shuffle(self.index_list)

    def index_to_color(self, index: int) -> Tuple[int, int, int]:
        """Get color from index."""
        selected_index = index % len(self.colors)
        selected_color = self.colors[self.index_list[selected_index]]
        return selected_color


class ImageLabeler:
    """Class for handeling the gui for labeling images."""

    def __init__(self, image_manager: ImageManager, dataset_manager: DatasetManager):
        """Initialize the image labeler."""
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
        self.buttons = self.dataset_manager.class_description.class_names
        self.current_class = 0
        self.color_manager = ColorManager()
        self.class_counts = [0 for _ in range(len(self.buttons))]
        self.menu_scroll_offset = 0

    def apply_transform(self, pos: Tuple[int, int]) -> Tuple[float, float]:
        """
        Preforms transformation from screen to image coordinates.

        Args:
            pos: The position on the screen.
        Returns:
            The position on the image
        """
        x, y = pos
        return (x - self.offset[0]) / self.scale, (y - self.offset[1]) / self.scale

    def initialize_image_pos(self):
        """Initialize the scale and offset of the image."""
        self.scale = 1.0
        image = self.image_manager.load_image()
        self.offset = [
            (self.screen_size[0] - self.menu_width) / 2 - image.shape[1] / 2,
            (self.screen_size[1] - self.menu_height) / 2 - image.shape[0] / 2,
        ]

    def event(self):
        """Handle events."""
        self.mouse_pos = pg.mouse.get_pos()

        for event in pg.event.get():
            if event.type == pg.QUIT:
                self.running = False

            if self.mouse_pos[0] > self.screen_size[0] - self.menu_width:
                if event.type == pg.MOUSEBUTTONDOWN and event.button == 1:
                    if self.mouse_pos[1] < self.screen_size[1] - 70:
                        self.check_button_click(self.mouse_pos)
                    self.check_navigation_click(self.mouse_pos)
                if event.type == pg.MOUSEWHEEL:
                    self.menu_scroll_offset -= event.y * 20
                    self.menu_scroll_offset = max(0, self.menu_scroll_offset)
                    self.menu_scroll_offset = min(self.menu_scroll_offset, len(self.buttons) * 60 - self.screen_size[1] + 70)
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
        """Check if a button was clicked."""
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
        """Check if a navigation button was clicked."""
        button_width = 100
        button_height = 50
        prev_button_rect = pg.Rect(self.screen_size[0] - self.menu_width + 10, self.screen_size[1] - button_height - 10, button_width, button_height)
        next_button_rect = pg.Rect(self.screen_size[0] - button_width - 10, self.screen_size[1] - button_height - 10, button_width, button_height)

        if prev_button_rect.collidepoint(pos):
            self.dataset_manager.remove_last_image()
            self.image_manager.previous_image()

        elif next_button_rect.collidepoint(pos):
            self.dataset_manager.save_image(self.image_manager.load_image(), self.bounding_boxes)
            self.bounding_boxes = []
            self.image_manager.next_image()
            self.initialize_image_pos()

    def is_mouse_on_image(self, pos: Tuple[int, int]) -> bool:
        """Return True if the mouse is on the image."""
        image = self.image_manager.load_image()
        left_bound = self.offset[0]
        right_bound = self.offset[0] + image.shape[1] * self.scale
        upper_bound = self.offset[1]
        lower_bound = self.offset[1] + image.shape[0] * self.scale
        left_right = left_bound <= pos[0] <= right_bound
        upper_lower = upper_bound <= pos[1] <= lower_bound
        return left_right and upper_lower

    def clamp_to_image_bounds(self, pos: Tuple[float, float]) -> Tuple[float, float]:
        """Clamp the position to the image bounds."""
        image = self.image_manager.load_image()

        return (
            max(0, min(image.shape[1], pos[0])),
            max(0, min(image.shape[0], pos[1]))
        )

    def draw(self):
        """Draw the image and bounding boxes."""
        self.screen.fill((255, 255, 255))

        image = self.image_manager.load_image()  # Assuming NumPy array (H, W, 3)
        img_height, img_width = image.shape[:2]
        # Calculate visible area in the image's coordinate system
        left = max(0, int((0 - self.offset[0]) / self.scale))
        top = max(0, int((0 - self.offset[1]) / self.scale))
        right = min(img_width, int((self.screen_size[0] - self.offset[0]) / self.scale + 1))
        bottom = min(img_height, int((self.screen_size[1] - self.offset[1]) / self.scale + 1))
        # Crop the visible region of the image
        visible_image = image[top:bottom, left:right]
        # Convert cropped image to Pygame surface
        if visible_image.size > 0:  # Ensure there's something to draw
            visible_surface = pg.surfarray.make_surface(visible_image.transpose((1, 0, 2)))
            visible_surface = pg.transform.scale(
                visible_surface,
                (int((right - left) * self.scale), int((bottom - top) * self.scale))
            )
            # Draw the cropped and scaled image at the correct offset
            self.screen.blit(visible_surface, (
                self.offset[0] + left * self.scale,
                self.offset[1] + top * self.scale
            ))
        for box in self.bounding_boxes:
            rect = (
                int(box.x_min * self.scale + self.offset[0]),
                int(box.y_min * self.scale + self.offset[1]),
                int((box.x_max - box.x_min) * self.scale),
                int((box.y_max - box.y_min) * self.scale),
            )
            color = self.color_manager.index_to_color(box.class_id)
            pg.draw.rect(self.screen, color, rect, 2)

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
            text_surface = font.render(f'{i} {label} ({self.class_counts[i]})', True, (0, 0, 0))
            text_rect = text_surface.get_rect(center=button_rect.center)
            self.screen.blit(text_surface, text_rect)

        button_width = 100
        button_height = 50

        pg.draw.rect(self.screen, (180, 180, 180), (self.screen_size[0] - self.menu_width, self.screen_size[1] - button_height - 20, self.menu_width, button_height + 20))

        prev_button_rect = pg.Rect(self.screen_size[0] - self.menu_width + 10, self.screen_size[1] - button_height - 10, button_width, button_height)
        next_button_rect = pg.Rect(self.screen_size[0] - button_width - 10, self.screen_size[1] - button_height - 10, button_width, button_height)

        pg.draw.rect(self.screen, (150, 150, 250), prev_button_rect)
        pg.draw.rect(self.screen, (150, 150, 250), next_button_rect)

        prev_text = pg.font.SysFont(None, 24).render('Previous', True, (0, 0, 0))
        next_text = pg.font.SysFont(None, 24).render('Next', True, (0, 0, 0))

        self.screen.blit(prev_text, prev_text.get_rect(center=prev_button_rect.center))
        self.screen.blit(next_text, next_text.get_rect(center=next_button_rect.center))

        pg.display.flip()

    def loop(self):
        """Preforms the main loop."""
        while self.running:
            self.event()
            self.draw()
        pg.quit()

import pygame
from pygame import Surface


def init_display():
    pygame.init()
    size = (800, 480)  # standard 5" Raspberry Pi screen
    screen = pygame.display.set_mode(size)
    pygame.display.set_caption("Robot Eyes")
    return screen


def load_images():
    images = {}
    for name in ["neutral", "happy", "sad", "blink1", "blink2", "talk1", "talk2", "talk3"]:
        images[name] = pygame.image.load(f"robot_eyes/assets/{name}.png").convert_alpha()
    return images


def draw_expression(screen: Surface, images: dict, expression: str):
    screen.fill((0, 0, 0))
    screen.blit(images[expression], (0, 0))
    pygame.display.update()


def pump_events() -> bool:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            return False
    return True

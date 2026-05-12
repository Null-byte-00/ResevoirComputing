import pygame 
from core import RecurrentNetwork

def value_to_brightness(value, min_val, max_val):
    """
    Converts a neuron value into brightness from 0 to 255.
    """
    if max_val - min_val < 1e-8:
        return 0

    normalized = (value - min_val) / (max_val - min_val)
    normalized = max(0.0, min(1.0, normalized))

    return int(normalized * 255)


def main():
    pygame.init()

    width, height = 1200, 500
    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption("Reservoir Computing Network Visualization")

    clock = pygame.time.Clock()

    num_neurons = 20
    network = RecurrentNetwork(
        num_neurons=num_neurons,
        timestep=0.001,
        oscillator_frequency=100,
        change_rate=0.01
    )

    running = True

    radius = 25
    spacing = width // (num_neurons + 1)
    y_pos = height // 2

    font = pygame.font.SysFont(None, 24)

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        state = network.step()

        values = state.detach().flatten()
        min_val = values.min().item()
        max_val = values.max().item()

        screen.fill((20, 20, 20))

        for i, value in enumerate(values):
            value = value.item()

            brightness = value_to_brightness(value, min_val, max_val)

            color = (brightness, brightness, brightness)

            x_pos = spacing * (i + 1)

            pygame.draw.circle(screen, color, (x_pos, y_pos), radius)

            label = font.render(str(i), True, (200, 200, 200))
            screen.blit(label, (x_pos - 6, y_pos + 40))

            value_text = font.render(f"{value:.2f}", True, (200, 200, 200))
            screen.blit(value_text, (x_pos - 20, y_pos - 55))

        pygame.display.flip()

        clock.tick(60)

    pygame.quit()


if __name__ == "__main__":
    main()
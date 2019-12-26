from window import Window,WindowConfig
from assets import Snake,Food


cfg = WindowConfig(width = 800,
                    height = 600,
                    player = Snake,
                    food = Food,
                    player_size = (20,20),
                    food_size = (18,18)
                    )

if __name__ == "__main__":
    game = Window(cfg)
    game.on_execute()
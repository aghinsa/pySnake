from window import Window,WindowConfig
from assets import Snake


cfg = WindowConfig(width = 800,
                    height = 600,
                    player = Snake
                    )

if __name__ == "__main__":
    game = Window(cfg)
    game.on_execute()
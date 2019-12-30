from game import Game,GameConfig
from assets import Snake,Food


cfg = GameConfig(width = 400,
                    height = 400,
                    player = Snake,
                    food = Food,
                    player_size = (20,20),
                    food_size = (20,20),
                    render = True
                    )
SNAKE_GAME = Game(cfg)
if __name__ == "__main__":
    SNAKE_GAME.on_execute()
from game import Game,GameConfig
from assets import Snake,Food


cfg = GameConfig(width = 800,
                    height = 600,
                    player = Snake,
                    food = Food,
                    player_size = (20,20),
                    food_size = (18,18)
                    )

if __name__ == "__main__":
    snake_game = Game(cfg)
    snake_game.on_execute()
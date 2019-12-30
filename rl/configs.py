STACK_SIZE = 2
GAMMA = 0.99
REPLAY_CAPACITY = 7500
BATCH_SIZE = 64
SHOW = False
MIN_REPLAY_HISTORY = REPLAY_CAPACITY #number of transitions that should be experienced
                    #before the agent begins training its value function
BASE_DIR = 'summaries/snake_classic/'
TARGET_UPDATE_PERIOD = 1000 # update period for the target network
SUMMARY_WRITING_FREQUENCY = 50
EVAL_MODE=False
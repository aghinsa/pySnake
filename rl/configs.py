STACK_SIZE = 4
GAMMA = 0.99
REPLAY_CAPACITY = 10000
BATCH_SIZE = 32
SHOW = False
MIN_REPLAY_HISTORY = 10000 #number of transitions that should be experienced
                    #before the agent begins training its value function
BASE_DIR = 'summaries/snake_classic/'
TARGET_UPDATE_PERIOD = 500 # update period for the target network
SUMMARY_WRITING_FREQUENCY = 500
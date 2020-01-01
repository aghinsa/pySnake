import tensorflow as tf
import pygame
from dopamine.discrete_domains.run_experiment import Runner
from dopamine.discrete_domains import run_experiment
from dopamine.utils import agent_visualizer
from dopamine.utils import atari_plotter
from dopamine.utils import bar_plotter
from dopamine.utils import line_plotter
from dopamine.utils import plotter

def infer_shape(x):
    x = tf.convert_to_tensor(x)

    # If unknown rank, return dynamic shape
    if x.shape.dims is None:
        return tf.shape(x)

    static_shape = x.shape.as_list()
    dynamic_shape = tf.shape(x)

    ret = []
    for i in range(len(static_shape)):
        dim = static_shape[i]
        if dim is None:
            dim = dynamic_shape[i]
        ret.append(dim)

    return ret

def merge_last_two_dims(tensor):
    shape = infer_shape(tensor)
    shape[-2] *= shape[-1]
    shape.pop(-1)
    return tf.reshape(tensor, shape)

class SnakeRunner(Runner):
    def visualize(self, record_path, num_global_steps=500):
        if not tf.gfile.Exists(record_path):
            tf.gfile.MakeDirs(record_path)
        self._agent.eval_mode = True

        # Set up the game playback rendering.
        atari_params = {
                'environment': self._environment,
                'width': 400,
                'height': 400,
                }
        atari_plot = atari_plotter.AtariPlotter(parameter_dict=atari_params)
        # Plot the rewards received next to it.
        reward_params = {'x': atari_plot.parameters['width'],
                        'xlabel': 'Timestep',
                        'ylabel': 'Reward',
                        'title': 'Rewards',
                        'get_line_data_fn': self._agent.get_rewards}
        reward_plot = line_plotter.LinePlotter(parameter_dict=reward_params)
        action_names = [
            'Action {}'.format(x) for x in range(self._agent.num_actions)]
        # Plot Q-values (DQN) or Q-value distributions (Rainbow).
        q_params = {'x': atari_plot.parameters['width'] // 2,
                'y': atari_plot.parameters['height'],
                'legend': action_names}
        if 'DQN' in self._agent.__class__.__name__:
            q_params['xlabel'] = 'Timestep'
            q_params['ylabel'] = 'Q-Value'
            q_params['title'] = 'Q-Values'
            q_params['get_line_data_fn'] = self._agent.get_q_values
            q_plot = line_plotter.LinePlotter(parameter_dict=q_params)
        else:
            q_params['xlabel'] = 'Return'
            q_params['ylabel'] = 'Return probability'
            q_params['title'] = 'Return distribution'
            q_params['get_bar_data_fn'] = self._agent.get_probabilities
            q_plot = bar_plotter.BarPlotter(parameter_dict=q_params)
        screen_width = (
            atari_plot.parameters['width'] + reward_plot.parameters['width'])
        screen_height = (
            atari_plot.parameters['height'] + q_plot.parameters['height'])
        # Dimensions need to be divisible by 2:
        if screen_width % 2 > 0:
            screen_width += 1
        if screen_height % 2 > 0:
            screen_height += 1
        visualizer = agent_visualizer.AgentVisualizer(
            record_path=record_path, plotters=[atari_plot, reward_plot, q_plot],
            screen_width=screen_width, screen_height=screen_height)

        global_step = 0
        while global_step < num_global_steps:
            initial_observation = self._environment.reset()
            action = self._agent.begin_episode(initial_observation)
            while True:
                observation, reward, is_terminal, _ = self._environment.step(action)
                global_step += 1
                visualizer.visualize()
                if self._environment.game_over or global_step >= num_global_steps:
                    break
                elif is_terminal:
                    self._agent.end_episode(reward)
                    action = self._agent.begin_episode(observation)
                else:
                    action = self._agent.step(reward, observation)
            self._end_episode(reward)
        visualizer.generate_video()

class SnakePlotter(plotter.Plotter):
    def __init__(self, parameter_dict=None):
        super().__init__(parameter_dict)
        assert 'environment' in self.parameters
        self.game_surface = pygame.Surface((self.parameters['width'],
                                            self.parameters['height']))

    def draw(self):
        """Render the Atari 2600 frame.

        Returns:
        object to be rendered by AgentVisualizer.
        """
        environment = self.parameters['environment']
        obs = environment.render(mode='rgb_array').astype(np.int32)

        return pygame.transform.scale(self.game_surface,
                                    (self.parameters['width'],
                                    self.parameters['height']))
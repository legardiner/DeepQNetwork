import numpy as np
import tensorflow as tf
from collections import deque


class BreakoutStateProcessor():
    """Given a state for the game breakout, return a preprocessed state

    Attributes:
        input_state (numpy.ndarray): Numpy array from OpenAI gym
            containing state information for each trajectory
        output (numpy.ndarray): Numpy array of shape (84, 84) containing
            scaled-down and preprocessed state information

    Return:
        output (numpy.ndarray)
    """
    def __init__(self):
        with tf.variable_scope("state_processor"):
            self.input_state = tf.placeholder(shape=[210, 160, 3],
                                              dtype=tf.uint8)
            self.output = tf.image.rgb_to_grayscale(self.input_state)
            self.output = tf.image.crop_to_bounding_box(
                self.output, 34, 0, 160, 160)
            self.output = tf.image.resize_images(
                self.output, [84, 84],
                method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
            self.output = tf.squeeze(self.output)

    def process(self, sess, state):
        return sess.run(self.output, {self.input_state: state})


class DQN():
    """Deep Q Neural Network with Huber Loss

    The fc2 returns Q values for every possible action. These Q values from the
    target network are used as the ground truth labels/y to optimize the
    evaluation network.

    The evaluation network takes the state, actions, and the target Q rewards.

    Convolutional layers may need to be reduced for breakout to prevent
    overfitting

    Args:
        learning_rate (float): learning rate for optimizer
        state_size (int): length of the stacked state vectors
        action_size (int): number of possible actions
        name (string): name of Deep Q Network object
    Attributes:
        inputs_ (numpy.ndarray): Numpy array of shape (state_size, )
            containing state information for each trajectory
        actions_ (numpy.ndarray): Numpy array of shape (action_size, )
            containg an one-hot encoding action vector for each trajectory
        target_Q (numpy.ndarray): Ground truth label from target network
        epoch_loss (float): Loss from evaluation network scalar for plotting
        avg_max_Q (float): Avg max q for batch for plotting
        total_game_reward (float): Total game reward for plotting
        scope (str): Name to distinguish target and evaluation networks
        Q (float): Q value prediction from the eval network given an action
    """
    def __init__(self, learning_rate=0.001, state_size=[84, 84, 4],
                 action_size=4, name="DQN"):
        with tf.variable_scope(name):
            self.inputs_ = tf.placeholder(
                tf.float32, [None, *state_size], name="inputs")
            self.actions_ = tf.placeholder(
                tf.float32, [None, action_size], name="actions")
            self.target_Q = tf.placeholder(
                tf.float32, [None], name="target_Q")
            self.epoch_loss = tf.placeholder(tf.float32, name="epoch_loss")
            self.avg_max_Q = tf.placeholder(tf.float32, name="avg_max_Q")
            self.total_game_reward = tf.placeholder(
                tf.float32, name="total_game_reward")
            self.scope = name
            self.scaled_inputs = self.inputs_ / 255.0
            self.conv1 = tf.layers.conv2d(
                inputs=self.scaled_inputs,
                filters=32,
                kernel_size=[8, 8],
                strides=[4, 4],
                padding="VALID",
                kernel_initializer=tf.variance_scaling_initializer(scale=2),
                activation=tf.nn.relu, use_bias=False, name='conv1'
            )
            self.conv2 = tf.layers.conv2d(
                inputs=self.conv1,
                filters=64,
                kernel_size=[4, 4],
                strides=[2, 2],
                padding="VALID",
                kernel_initializer=tf.variance_scaling_initializer(scale=2),
                activation=tf.nn.relu, use_bias=False, name='conv2'
            )
            self.conv3 = tf.layers.conv2d(
                inputs=self.conv2,
                filters=64,
                kernel_size=[3, 3],
                strides=[1, 1],
                padding="VALID",
                kernel_initializer=tf.variance_scaling_initializer(scale=2),
                activation=tf.nn.relu, use_bias=False, name='conv3'
            )
            self.flatten = tf.layers.flatten(self.conv3)
            self.fc1 = tf.layers.dense(
                self.flatten, 512, activation=tf.nn.relu,
                kernel_initializer=tf.variance_scaling_initializer(scale=2),
                name="fc1")
            self.fc2 = tf.layers.dense(
                self.fc1, units=action_size,
                kernel_initializer=tf.variance_scaling_initializer(scale=2),
                activation=tf.nn.relu)
        with tf.variable_scope("Q"):
            self.Q = tf.reduce_sum(tf.multiply(self.fc2, self.actions_),
                                   axis=1)
        with tf.variable_scope("loss"):
            self.losses = tf.losses.huber_loss(self.target_Q, self.Q)
            self.loss = tf.reduce_mean(self.losses)
        with tf.variable_scope("train"):
            self.optimizer = tf.train.RMSPropOptimizer(learning_rate,
                                                       momentum=0.95,
                                                       epsilon=0.01)
            self.train = self.optimizer.minimize(self.loss)
        with tf.variable_scope("summaries"):
            tf.summary.scalar("epoch_loss", self.epoch_loss)
            tf.summary.scalar("avg_max_Q", self.avg_max_Q)
            tf.summary.scalar("total_game_reward", self.total_game_reward)
            self.summary_op = tf.summary.merge_all()


class ReplayBuffer():
    """Replay buffer to hold game experiences

    Each experience seen is added to the replay buffer and
    replay buffer is sample each time the evaluation network
    is optimized

    Args:
        buffer_size (int): max number of memory to store
        batch_size (int): number of memories to sample
    """
    def __init__(self, buffer_size):
        self.buffer = deque(maxlen=buffer_size)

    def add(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        buffer_size = len(self.buffer)
        index = np.random.choice(np.arange(buffer_size),
                                 size=batch_size,
                                 replace=False)

        return [self.buffer[i] for i in index]


class ModelParametersCopier():
    """Copies parameters/weights from evaluation network to target network for
    a given tensorflow session

    Args:
        sess (tf.Session)
        evaluationNetwork (DQN)
        targetNetwork (DQN)
    """
    def __init__(self, evaluationNetwork, targetNetwork):
        evaluation_params = [t for t in tf.trainable_variables() if
                             t.name.startswith(evaluationNetwork.scope)]
        evaluation_params = sorted(evaluation_params, key=lambda v: v.name)
        target_params = [t for t in tf.trainable_variables() if
                         t.name.startswith(targetNetwork.scope)]
        target_params = sorted(target_params, key=lambda v: v.name)

        self.update_ops = []
        for evaluation_v, target_v in zip(evaluation_params, target_params):
            op = target_v.assign(evaluation_v)
            self.update_ops.append(op)

    def make(self, sess):
        sess.run(self.update_ops)


def preprocess_observation(obs):
    """Given a state from Pacman, return downsampled and preprocessed state

    Args:
        obs (numpy.ndarray): Numpy array from OpenAI gym
            containing state information for each trajectory

    Return:
        output (numpy.ndarray): Numpy array of shape (88, 80) containing
            scaled-down and preprocessed state information
    """
    mspacman_color = 210 + 164 + 74
    # Crop and downsize
    img = obs[1:176:2, ::2]
    # Convert to greyscale
    img = img.sum(axis=2)
    # Improve contrast
    img[img == mspacman_color] = 0
    # Normalize from -128 to 127
    img = (img // 3 - 128).astype(np.int8)
    return img.reshape(88, 80)


def stack_frames(stacked_frames, frame_dim, frame, stack_size, is_new_episode):
    """ Creates a stack of the four most recent frames

    Args:
        stacked_frames (list): List of previous frames
        frame_dim (tuple): State size for game
        stack_size (int): Number of frames to keep
        is_new_episode (boolean): New game indicator

    Returns:
        stacked_state (numpy.ndarray): Stacked frames for input to network
        stacked_frames (list): List of frames for next episode
    """
    if is_new_episode:
        # Clear our stacked_frames
        stacked_frames = [np.zeros(frame_dim, dtype=np.int)
                          for i in range(stack_size)]
        # Because we're in a new episode, copy the same frame 4x
        stacked_frames.append(frame)
        stacked_frames.append(frame)
        stacked_frames.append(frame)
        stacked_frames.append(frame)
        # Stack the frames
        stacked_state = np.dstack(stacked_frames[-4:])
    else:
        # Append frame to list
        stacked_frames.append(frame)
        # Build the stacked state from the last four frames
        stacked_state = np.stack(stacked_frames[-4:], axis=2)
    return stacked_state, stacked_frames[-4:]


def predict_action(decay_rate, decay_step, state, env, sess, evalNetwork):
    """
    This function will choose an action from state s by using epsilon greedy

    It has an exponential decay rate to decrease the likelihood of the
    network exploring vs exploiting over time

    Args:
        decay_rate (float): Exponential decay rate for epsilon greedy
        decay_step (int): Step within training network
        state (np.ndarray): Given state within a game
        env (gym.make()): OpenAI gym game for sample action space
        sess (tf.Session()): TensorFlow session
        evalNetwork (DQN): Deep Q Network
    """
    explore_start = 1.0
    explore_stop = 0.01
    explore_exploit_tradeoff = np.random.rand()
    explore_probability = explore_stop + (explore_start - explore_stop) \
        * np.exp(-decay_rate * decay_step)

    if (explore_probability > explore_exploit_tradeoff):
        # Make a random action (exploration)
        action = env.action_space.sample()
    else:
        # Get action from Q-network (exploitation)
        # Estimate the Qs values state
        feed = {evalNetwork.inputs_: state.reshape((1, *state.shape))}
        Qs = sess.run(evalNetwork.fc2, feed_dict=feed)

        # Take the action with the biggest Q value
        action = np.argmax(Qs)

    return action

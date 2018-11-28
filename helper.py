import numpy as np
import tensorflow as tf
from collections import deque

class StateProcessor():
    def __init__(self):
        with tf.variable_scope("state_processor"):
            self.input_state = tf.placeholder(shape=[210, 160, 3], dtype=tf.uint8)
            self.output = tf.image.rgb_to_grayscale(self.input_state)
            self.output = tf.image.crop_to_bounding_box(self.output, 34, 0, 160, 160)
            self.output = tf.image.resize_images(
                self.output, [84, 84], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
            self.output = tf.squeeze(self.output)
    def process(self, sess, state):
        return sess.run(self.output, { self.input_state: state })

class DQN():
    def __init__(self, learning_rate=0.001, state_size=[84,84,4], action_size=4, name="DQN"):
        with tf.variable_scope(name):
            self.inputs_ = tf.placeholder(
                tf.float32, [None, *state_size], name="inputs")
            self.actions_ = tf.placeholder(
                tf.float32, [None, action_size], name="actions")
            self.target_Q = tf.placeholder(
                tf.float32, [None], name="target_Q")
            self.avg_max_Q = tf.placeholder(tf.float32, name="avg_max_Q")
            self.total_game_reward = tf.placeholder(tf.float32, name="total_game_reward")
            self.scope = name
            self.scaled_inputs = self.inputs_ / 255.0
            self.conv1 = tf.layers.conv2d(
                inputs = self.scaled_inputs, 
                filters = 32,
                kernel_size = [8,8],
                strides = [4,4],
                padding = "VALID",
                kernel_initializer=tf.variance_scaling_initializer(scale=2),
                activation=tf.nn.relu, use_bias=False, name='conv1'
            )
            self.conv2 = tf.layers.conv2d(
                inputs = self.inputs_, 
                filters = 64,
                kernel_size = [4,4],
                strides = [2,2],
                padding = "VALID",
                kernel_initializer=tf.variance_scaling_initializer(scale=2),
                activation=tf.nn.relu, use_bias=False, name='conv2'
            )
            self.conv3 = tf.layers.conv2d(
                inputs = self.inputs_, 
                filters = 128,
                kernel_size = [4,4],
                strides = [2,2],
                padding = "VALID",
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
                activation=None)
        with tf.variable_scope("Q"):
            self.Q = tf.reduce_sum(tf.multiply(self.fc2, self.actions_), axis=1)
        with tf.variable_scope("loss"):
            self.losses = tf.losses.huber_loss(self.target_Q, self.Q)
            self.loss = tf.reduce_mean(self.losses)
        with tf.variable_scope("train"):
            self.optimizer = tf.train.AdamOptimizer(learning_rate)
            self.train = self.optimizer.minimize(self.losses)
        with tf.variable_scope("summaries"):
            # tf.summary.scalar("loss", self.loss)
            tf.summary.scalar("avg_max_Q", self.avg_max_Q)
            tf.summary.scalar("total_game_reward", self.total_game_reward)
            self.summary_op = tf.summary.merge_all()

def stack_frames(stacked_frames, frame, stack_size, is_new_episode):    
    if is_new_episode:
        # Clear our stacked_frames
        stacked_frames = deque([np.zeros((84,84), dtype=np.int) for i in range(stack_size)], maxlen=4)
        
        # Because we're in a new episode, copy the same frame 4x
        stacked_frames.append(frame)
        stacked_frames.append(frame)
        stacked_frames.append(frame)
        stacked_frames.append(frame)
        
        # Stack the frames
        stacked_state = np.stack(stacked_frames, axis=2)
        
    else:
        # Append frame to deque, automatically removes the oldest frame
        stacked_frames.append(frame)

        # Build the stacked state (first dimension specifies different frames)
        stacked_state = np.stack(stacked_frames, axis=2) 
    
    return stacked_state, stacked_frames

class ModelParametersCopier():    
    def __init__(self, evaluationNetwork, targetNetwork):
        evaluation_params = [t for t in tf.trainable_variables() if t.name.startswith(evaluationNetwork.scope)]
        evaluation_params = sorted(evaluation_params, key=lambda v: v.name)
        target_params = [t for t in tf.trainable_variables() if t.name.startswith(targetNetwork.scope)]
        target_params = sorted(target_params, key=lambda v: v.name)

        self.update_ops = []
        for evaluation_v, target_v in zip(evaluation_params, target_params):
            op = target_v.assign(evaluation_v)
            self.update_ops.append(op)
            
    def make(self, sess):
        sess.run(self.update_ops)

class ReplayBuffer():
    def __init__(self, buffer_size):
        self.buffer = deque(maxlen = buffer_size)
    
    def add(self, experience):
        self.buffer.append(experience)
    
    def sample(self, batch_size):
        buffer_size = len(self.buffer)
        index = np.random.choice(np.arange(buffer_size),
                                size = batch_size,
                                replace = False)
        
        return [self.buffer[i] for i in index]

"""
This function will do the part
With ϵ select a random action atat, otherwise select at=argmaxaQ(st,a)
"""
def predict_action(decay_rate, decay_step, state, env, sess, evaluationNetwork):
    ## EPSILON GREEDY STRATEGY
    # Choose action a from state s using epsilon greedy.
    ## First we randomize a number
    explore_start = 1.0
    explore_stop = 0.01
    exp_exp_tradeoff = np.random.rand()

    # Here we'll use an improved version of our epsilon greedy strategy used in Q-learning notebook
    explore_probability = explore_stop + (explore_start - explore_stop) * np.exp(-decay_rate * decay_step)
    
    if (explore_probability > exp_exp_tradeoff):
        # Make a random action (exploration)
        action = env.action_space.sample()
    else:
        # Get action from Q-network (exploitation)
        # Estimate the Qs values state
        Qs = sess.run(evaluationNetwork.fc2, feed_dict = {evaluationNetwork.inputs_: state.reshape((1, *state.shape))})
        
        # Take the biggest Q value (= the best action)
        action = np.argmax(Qs)
                
    return action, explore_probability
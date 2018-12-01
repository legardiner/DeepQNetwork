import gym
import gym.spaces
import argparse
import numpy as np
import tensorflow as tf
from helper import StateProcessor, DQN, stack_frames, ReplayBuffer, predict_action
from collections import deque
import logging
import os

parser = argparse.ArgumentParser(description='Deep Q Network reinforcement \
                                 learning model for breakout game')
parser.add_argument('--learning_rate', default=0.001, help='Learning rate for \
                    optimizer')
parser.add_argument('--discount_rate', default=0.95, help='Discount rate for \
                    future rewards')
parser.add_argument('--num_games', default=1000, help='Number of games to play')
parser.add_argument('--stack_size', default=4, help='Number of frames to stack')
parser.add_argument('--state_size', default=[88, 80, 4], help='Number of state values')
parser.add_argument('--action_size', default=4, help='Number of actions')
parser.add_argument('--output_size', default=1, help='Number of output neurons \
                    for value function network')
parser.add_argument('--buffer_size', default=1000000, help='Max number of memories in the replay buffer')
parser.add_argument('--batch_size', default=64, help='Number of memories to sample from the replay buffer')
parser.add_argument('--decay_rate', default=0.00001, help='Exponential decay rate for epsilon greedy')
parser.add_argument('--log_dir', default='logs/mspacman/', help='Path to directory for logs for \
                    tensorboard visualization')
parser.add_argument('--model_path', default='mspacman/model5_1350', help='Path to model checkpoint')
parser.add_argument('--run_num', required=True, help='Provide a run number to correctly log')

def main(args):
    stack_size = int(args.stack_size)
    decay_rate = float(args.decay_rate)
    batch_size = int(args.batch_size)
    # Load game
    env = gym.make("MsPacman-v0")
    # Initialize the game
    state = env.reset()
    # Reset tensorflow graph
    tf.reset_default_graph()
    # Initialize both networks
    evaluationNetwork = DQN(learning_rate=args.learning_rate,
                             state_size=args.state_size,
                             action_size=args.action_size,
                            name="evaluationNetwork")
    targetNetwork = DQN(learning_rate=args.learning_rate,
                             state_size=args.state_size,
                             action_size=args.action_size,
                            name="targetNetwork")

    # Initialize image preprocessor
    state_processor = StateProcessor()
    # Initalize trainer
    saver = tf.train.Saver()
    # Initialize step counter
    step = 0
    # Initialize replay buffer
    memory = ReplayBuffer(args.buffer_size)
    # Create directory for logs
    if not os.path.exists(os.path.join(args.log_dir, args.run_num)):
        logging.info("Creating directory {0}".format(os.path.join(args.log_dir, args.run_num)))
        os.mkdir(os.path.join(args.log_dir, args.run_num))
        os.mkdir(os.path.join(args.log_dir, args.run_num, "eval"))
    # Start tensorflow
    with tf.Session() as sess:
        writer = tf.summary.FileWriter(os.path.join(args.log_dir, args.run_num, "eval"), sess.graph)
        sess.run(tf.global_variables_initializer())
        saver.restore(sess, args.model_path)
        # Fill buffer
        # Initialize a new game
        state = env.reset()
        for game in range(int(args.num_games)):
            # Initialize a new game
            state = env.reset()
            # Initialize rewards for the episode
            game_rewards = []
            # Initialize deque with zero-images one array for each image
            stacked_frames = [np.zeros((88,80), dtype=np.int) for i in range(stack_size)]
            # Preprocess and stack frames
            state = state_processor.process(sess, state)
            state, stacked_frames = stack_frames(stacked_frames, (88, 80), state, stack_size, True)
            # Run the game/episode until it's done
            while True:
                # Take action based on epsilon greedy
                step += 1
                action, explore_probability = predict_action(decay_rate, step, state, env, sess, evaluationNetwork)
                next_state, reward, done, _ = env.step(action)
                # Create one hot encoding for action for network input
                one_hot_action = np.zeros(int(args.action_size))
                one_hot_action[action] = 1

                # Track reward at each step
                game_rewards.append(reward)

                if done:
                    next_state = np.zeros((88,80), dtype=np.int)
                    next_state, stacked_frames = stack_frames(stacked_frames, (88, 80), next_state, stack_size, False)
                    memory.add((state, one_hot_action, reward, next_state, done))
                    break
                else:
                    next_state = state_processor.process(sess, next_state)
                    next_state, stacked_frames = stack_frames(stacked_frames, (88, 80), next_state, stack_size, False)
                    memory.add((state, one_hot_action, reward, next_state, done))
                    state = next_state

            # After iterating through all frames in a game, calculate the total game reward 
            total_game_reward = sum(game_rewards)
            logging.info("Game {0} Total Reward:\t{1}".format(game, total_game_reward))
    

if __name__ == "__main__":
    log_fmt = '%(asctime)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    logger = logging.getLogger(__name__)

    args = parser.parse_args()
    main(args)
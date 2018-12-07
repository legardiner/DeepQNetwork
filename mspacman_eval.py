import gym
import gym.spaces
import argparse
import numpy as np
import tensorflow as tf
import logging
import os
import matplotlib.pyplot as plt
from helper import preprocess_observation, DQN, ReplayBuffer, stack_frames, \
    predict_action

parser = argparse.ArgumentParser(description='Evaluator for Deep Q Network reinforcement \
                                 learning model for Ms. Pacman game')
parser.add_argument('--learning_rate', default=0.0001,
                    help='Learning rate for optimizer')
parser.add_argument('--num_games', default=1000,
                    help='Number of games to play')
parser.add_argument('--stack_size', default=4,
                    help='Number of frames to stack')
parser.add_argument('--state_size', default=(88, 80), help='State size')
parser.add_argument('--action_size', default=9, help='Number of game actions')
parser.add_argument('--decay_rate', default=0.00001,
                    help='Exponential decay rate for epsilon greedy')
parser.add_argument('--model_path', default='mspacman/model/model6_2290.ckpt',
                    help='Path to model checkpoint')


def main(args):
    # Parse non-string args for repeated use
    stack_size = int(args.stack_size)
    decay_rate = float(args.decay_rate)
    learning_rate = float(args.learning_rate)
    state_size = (int(args.state_size[0]), int(args.state_size[1]))
    stacked_state_size = [state_size[0], state_size[1],
                          stack_size]
    # Load and initalize game
    env = gym.make("MsPacman-v0")
    state = env.reset()

    # Reset tensorflow graph
    tf.reset_default_graph()

    # Initialize both networks
    evaluationNetwork = DQN(learning_rate=learning_rate,
                            state_size=stacked_state_size,
                            action_size=args.action_size,
                            name="evaluationNetwork")
    targetNetwork = DQN(learning_rate=learning_rate,
                        state_size=stacked_state_size,
                        action_size=args.action_size,
                        name="targetNetwork")

    # Initalize trainer
    saver = tf.train.Saver()
    # Initialize step counter
    step = 0

    # Start tensorflow
    with tf.Session() as sess:
        # Reload model
        sess.run(tf.global_variables_initializer())
        saver.restore(sess, args.model_path)

        rewards = []

        # Play games
        for game in range(int(args.num_games)):
            # Initialize a new game
            state = env.reset()

            # Initialize rewards for the episode
            game_rewards = []

            # Preprocess and stack frames
            stacked_frames = [np.zeros(state_size, dtype=np.int)
                              for i in range(stack_size)]
            state = preprocess_observation(state)
            state, stacked_frames = stack_frames(stacked_frames, state_size,
                                                 state, stack_size, True)
            # Run the game/episode until it's done
            while True:
                # Take action based on epsilon greedy
                step += 1
                action = predict_action(decay_rate, step, state, env, sess,
                                        evaluationNetwork)
                next_state, reward, done, _ = env.step(action)
                # Create one hot encoding for action for network input
                one_hot_action = np.zeros(int(args.action_size))
                one_hot_action[action] = 1

                # Track reward at each step
                game_rewards.append(reward)

                # Break if game is over or continue playing
                if done:
                    next_state = np.zeros(state_size, dtype=np.int)
                    next_state, stacked_frames = stack_frames(stacked_frames,
                                                              state_size,
                                                              next_state,
                                                              stack_size,
                                                              False)
                    break
                else:
                    next_state = preprocess_observation(next_state)
                    next_state, stacked_frames = stack_frames(stacked_frames,
                                                              state_size,
                                                              next_state,
                                                              stack_size,
                                                              False)
                    state = next_state

            # Calculate the total game reward
            total_game_reward = sum(game_rewards)
            rewards.append(total_game_reward)
            logging.info("Game {0} Total Reward:\t{1}".
                         format(game, total_game_reward))

    plt.plot(rewards)
    plt.savefig('mspacman.png')

if __name__ == "__main__":
    log_fmt = '%(asctime)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    logger = logging.getLogger(__name__)

    args = parser.parse_args()
    main(args)

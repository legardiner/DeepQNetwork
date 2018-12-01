import gym
import gym.spaces
import tensorflow as tf
import numpy as np
import argparse
import logging
import os
from collections import deque
from helper import StateProcessor, DQN, stack_frames, ModelParametersCopier, ReplayBuffer, predict_action

parser = argparse.ArgumentParser(description='Deep Q Network reinforcement \
                                 learning model for breakout game')
parser.add_argument('--learning_rate', default=0.001, help='Learning rate for \
                    optimizer')
parser.add_argument('--discount_rate', default=0.95, help='Discount rate for \
                    future rewards')
parser.add_argument('--epochs', default=100000, help='Number of epochs to train')
parser.add_argument('--stack_size', default=4, help='Number of frames to stack')
parser.add_argument('--state_size', default=[84, 84, 4], help='Number of state values')
parser.add_argument('--action_size', default=4, help='Number of actions')
parser.add_argument('--output_size', default=1, help='Number of output neurons \
                    for value function network')
parser.add_argument('--buffer_size', default=1000000, help='Max number of memories in the replay buffer')
parser.add_argument('--batch_size', default=64, help='Number of memories to sample from the replay buffer')
parser.add_argument('--decay_rate', default=0.00001, help='Exponential decay rate for epsilon greedy')
parser.add_argument('--log_dir', default='logs/breakout/', help='Path to directory for logs for \
                    tensorboard visualization')
parser.add_argument('--run_num', required=True, help='Provide a run number to correctly log')

def main(args):
    stack_size = int(args.stack_size)
    decay_rate = float(args.decay_rate)
    batch_size = int(args.batch_size)
    # Load game
    env = gym.make("Breakout-v0")
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
    # Initialize parameter updater
    update_parameters = ModelParametersCopier(evaluationNetwork, targetNetwork)
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
    # Start tensorflow
    with tf.Session() as sess:
        writer = tf.summary.FileWriter(os.path.join(args.log_dir, args.run_num), sess.graph)
        sess.run(tf.global_variables_initializer())
        # Fill buffer
        # Initialize a new game
        state = env.reset()
        # Initialize deque with zero-images one array for each image
        stacked_frames = [np.zeros((84,84), dtype=np.int) for i in range(stack_size)]
        # Preprocess and stack frames
        state = state_processor.process(sess, state)
        state, stacked_frames = stack_frames(stacked_frames, (84, 84), state, stack_size, True)
        # Run the game/episode until it's done
        while True:
            # Take action based on epsilon greedy
            step += 1
            action, explore_probability = predict_action(decay_rate, step, state, env, sess, evaluationNetwork)
    #         print("Step: {0}\tExplore Probability: {1}".format(step, explore_probability))
            next_state, reward, done, _ = env.step(action)
            # Create one hot encoding for action for network input
            one_hot_action = np.zeros(int(args.action_size))
            one_hot_action[action] = 1
            if done:
                next_state = np.zeros((84,84), dtype=np.int)
                next_state, stacked_frames = stack_frames(stacked_frames, (84, 84), next_state, stack_size, False)
                memory.add((state, one_hot_action, reward, next_state, done))
                break
            else:
                next_state = state_processor.process(sess, next_state)
                next_state, stacked_frames = stack_frames(stacked_frames, (84, 84), next_state, stack_size, False)
                memory.add((state, one_hot_action, reward, next_state, done))
                state = next_state

        # Start training
        for epoch in range(int(args.epochs)):
            # Initialize a new game
            state = env.reset()
            # Initialize rewards for the episode
            game_rewards = []
            # Initialize max Q values for the episode
            max_Q_values = []
            # Initialize deque with zero-images one array for each image
            stacked_frames = [np.zeros((84,84), dtype=np.int) for i in range(stack_size)]
            # Preprocess and stack frames
            state = state_processor.process(sess, state)
            state, stacked_frames = stack_frames(stacked_frames, (84, 84), state, stack_size, True)
            # Run the game/episode until it's done
            while True:
                # Take action based on epsilon greedy
                step += 1
                action, explore_probability = predict_action(decay_rate, step, state, env, sess, evaluationNetwork)
    #             print("Step: {0}\tExplore Probability: {1}".format(step, explore_probability))
                next_state, reward, done, _ = env.step(action)
                # Create one hot encoding for action for network input
                one_hot_action = np.zeros(int(args.action_size))
                one_hot_action[action] = 1

                # Track reward at each step
                game_rewards.append(reward)

                # Sample from memory buffer to feed into evaluation network
                batch = memory.sample(batch_size)
                batch_states = np.array([memory[0] for memory in batch], ndmin=4)
                batch_actions = np.array([memory[1] for memory in batch])
                batch_rewards = np.array([memory[2] for memory in batch]) 
                batch_next_states = np.array([memory[3] for memory in batch], ndmin=4)
                batch_dones = np.array([memory[4] for memory in batch])

                # Get ground truth from target network
                batch_target_Qs = []
                target_Qs = sess.run(targetNetwork.fc2, feed_dict = {targetNetwork.inputs_: batch_next_states})

                for i in range(batch_size):
                    terminal = batch_dones[i]
                    if terminal:
                        batch_target_Qs.append(batch_rewards[i])
                    else:
                        target = batch_rewards[i] + args.discount_rate * np.max(target_Qs[i])
                        batch_target_Qs.append(target)

                batch_targets = np.array([memory for memory in batch_target_Qs])

                # Update evaluation network
                loss, _ = sess.run([evaluationNetwork.loss, evaluationNetwork.train],
                                    feed_dict={evaluationNetwork.inputs_: batch_states,
                                                evaluationNetwork.actions_: batch_actions,
                                                evaluationNetwork.target_Q: batch_targets})


                # Get max Q for buffer for plotting
                predicted_Qs = sess.run(evaluationNetwork.fc2, feed_dict = {evaluationNetwork.inputs_: batch_states})
                max_Q_values.append(np.max(predicted_Qs))

                if done:
                    next_state = np.zeros((84,84), dtype=np.int)
                    next_state, stacked_frames = stack_frames(stacked_frames, (84, 84), next_state, stack_size, False)
                    memory.add((state, one_hot_action, reward, next_state, done))
                    break
                else:
                    next_state = state_processor.process(sess, next_state)
                    next_state, stacked_frames = stack_frames(stacked_frames, (84, 84), next_state, stack_size, False)
                    memory.add((state, one_hot_action, reward, next_state, done))
                    state = next_state
            # After iterating through all episodes in an epoch, calculate and store the average total reward 
            total_game_reward = sum(game_rewards)
            avg_max_Q = np.mean(max_Q_values)

            summary = sess.run(evaluationNetwork.summary_op, feed_dict={evaluationNetwork.epoch_loss: loss,
                                                                        evaluationNetwork.avg_max_Q: avg_max_Q, 
                                                                    evaluationNetwork.total_game_reward: total_game_reward})

            
            # Log and save models
            logging.info("Epoch: {0}\tAvg Max Q: {1}\tTotal Reward: {2}".format(epoch, avg_max_Q, total_game_reward))
            writer.add_summary(summary, epoch)
            if epoch % 10 == 0:
                    saver.save(sess, "./breakout/model{0}_{1}.ckpt".format(run_num, epoch))
                    print("Model Saved")

            if step % 10000 == 0:
                update_parameters.make(sess)
                logging.info("\nCopied model parameters to target network.")

if __name__ == "__main__":
    log_fmt = '%(asctime)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    logger = logging.getLogger(__name__)

    args = parser.parse_args()
    main(args)

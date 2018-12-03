import gym
import gym.spaces
import tensorflow as tf
import numpy as np
import argparse
import logging
import os
from helper import BreakoutStateProcessor, DQN, ReplayBuffer, \
    ModelParametersCopier, stack_frames, predict_action

parser = argparse.ArgumentParser(description='Deep Q Network reinforcement \
                                    learning model for breakout game')
parser.add_argument('--learning_rate', default=0.0001,
                    help='Learning rate for optimizer')
parser.add_argument('--discount_rate', default=0.95,
                    help='Discount rate for future rewards')
parser.add_argument('--epochs', default=100000,
                    help='Number of epochs to train')
parser.add_argument('--stack_size', default=4,
                    help='Number of frames to stack')
parser.add_argument('--state_size', default=(84, 84), help='State size')
parser.add_argument('--action_size', default=4, help='Number of game actions')
parser.add_argument('--buffer_size', default=1000000,
                    help='Max number of memories in the replay buffer')
parser.add_argument('--batch_size', default=64,
                    help='Number of memories to sample from the replay buffer')
parser.add_argument('--decay_rate', default=0.00001,
                    help='Exponential decay rate for epsilon greedy')
parser.add_argument('--log_dir', default='breakout/logs',
                    help='Path to directory for logs for tensorboard')
parser.add_argument('--model_dir', default='breakout/model',
                    help='Path to directory for model checkpoints')
parser.add_argument('--run_num', required=True,
                    help='Provide a run number to separate log files')


def main(args):
    # Parse non-string args for repeated use
    stack_size = int(args.stack_size)
    decay_rate = float(args.decay_rate)
    learning_rate = float(args.learning_rate)
    batch_size = int(args.batch_size)
    state_size = (int(args.state_size[0]), int(args.state_size[1]))
    stacked_state_size = [state_size[0], state_size[1],
                          stack_size]

    # Load and initalize game
    env = gym.make("Breakout-v0")
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

    # Initialize helper classes
    state_processor = BreakoutStateProcessor()
    memory = ReplayBuffer(args.buffer_size)
    update_parameters = ModelParametersCopier(evaluationNetwork, targetNetwork)

    saver = tf.train.Saver()
    # Initialize step counter for model updates and greedy epsilon
    step = 0

    # Create directory for logs
    log_dir_path = os.path.join(args.log_dir, args.run_num)
    if not os.path.exists(log_dir_path):
        logging.info("Creating directory {0}".format(log_dir_path))
        os.mkdir(log_dir_path)

    # Start tensorflow
    with tf.Session() as sess:
        writer = tf.summary.FileWriter(log_dir_path, sess.graph)
        sess.run(tf.global_variables_initializer())

        # -------------------- FILL BUFFER WITH MEMORIES -------------------
        # Initialize a new game
        state = env.reset()
        # Preprocess and stack frames
        stacked_frames = [np.zeros(state_size, dtype=np.int)
                          for i in range(stack_size)]
        state = state_processor.process(sess, state)
        state, stacked_frames = stack_frames(stacked_frames, state_size, state,
                                             stack_size, True)

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

            # Break if game is over or continue playing
            if done:
                next_state = np.zeros(state_size, dtype=np.int)
                next_state, stacked_frames = stack_frames(stacked_frames,
                                                          state_size,
                                                          next_state,
                                                          stack_size,
                                                          False)
                memory.add((state, one_hot_action, reward, next_state, done))
                break
            else:
                next_state = state_processor.process(sess, next_state)
                next_state, stacked_frames = stack_frames(stacked_frames,
                                                          state_size,
                                                          next_state,
                                                          stack_size,
                                                          False)
                memory.add((state, one_hot_action, reward, next_state, done))
                state = next_state

        # --------------------------- START TRAINING --------------------------
        for epoch in range(int(args.epochs)):
            # Initialize a new game
            state = env.reset()

            # Initialize lists for plotting
            game_rewards = []
            max_Q_values = []

            # Preprocess and stack frames
            stacked_frames = [np.zeros(state_size, dtype=np.int)
                              for i in range(stack_size)]
            state = state_processor.process(sess, state)
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

                # Sample from memory buffer to feed into evaluation network
                batch = memory.sample(batch_size)
                batch_states = np.array([memory[0] for memory in batch],
                                        ndmin=4)
                batch_actions = np.array([memory[1] for memory in batch])
                batch_rewards = np.array([memory[2] for memory in batch])
                batch_next_states = np.array([memory[3] for memory in batch],
                                             ndmin=4)
                batch_dones = np.array([memory[4] for memory in batch])

                # Get ground truth from target network
                batch_target_Qs = []
                feed = {targetNetwork.inputs_: batch_next_states}
                target_Qs = sess.run(targetNetwork.fc2, feed_dict=feed)

                # Calculate discounted Qs for batch
                for i in range(batch_size):
                    terminal = batch_dones[i]
                    # If the game is over, take the final reward
                    if terminal:
                        batch_target_Qs.append(batch_rewards[i])
                    else:
                        target = batch_rewards[i] + args.discount_rate * \
                                 np.max(target_Qs[i])
                        batch_target_Qs.append(target)
                batch_target = np.array([memory for memory in batch_target_Qs])

                # Update evaluation network
                feed = {evaluationNetwork.inputs_: batch_states,
                        evaluationNetwork.actions_: batch_actions,
                        evaluationNetwork.target_Q: batch_target}
                loss, _ = sess.run([evaluationNetwork.loss,
                                    evaluationNetwork.train],
                                   feed_dict=feed)

                # Get max Q for batch for plotting
                feed = {evaluationNetwork.inputs_: batch_states}
                predicted_Qs = sess.run(evaluationNetwork.fc2,
                                        feed_dict=feed)
                max_Q_values.append(np.max(predicted_Qs))

                # Break if game is over or continue playing
                if done:
                    next_state = np.zeros(state_size, dtype=np.int)
                    next_state, stacked_frames = stack_frames(stacked_frames,
                                                              state_size,
                                                              next_state,
                                                              stack_size,
                                                              False)
                    memory.add((state, one_hot_action, reward, next_state,
                                done))
                    break
                else:
                    next_state = state_processor.process(sess, next_state)
                    next_state, stacked_frames = stack_frames(stacked_frames,
                                                              state_size,
                                                              next_state,
                                                              stack_size,
                                                              False)
                    memory.add((state, one_hot_action, reward, next_state,
                                done))
                    state = next_state

            # After playing entire game, plot game rewards and max Qs
            total_game_reward = sum(game_rewards)
            avg_max_Q = np.mean(max_Q_values)

            feed = {evaluationNetwork.epoch_loss: loss,
                    evaluationNetwork.avg_max_Q: avg_max_Q,
                    evaluationNetwork.total_game_reward: total_game_reward}
            summary = sess.run(evaluationNetwork.summary_op, feed_dict=feed)

            # Log and save models
            logging.info("Epoch: {0}\tAvg Max Q: {1}\tTotal Reward: {2}".
                         format(epoch, avg_max_Q, total_game_reward))
            writer.add_summary(summary, epoch)
            if epoch % 10 == 0:
                    saver.save(sess, "{0}/model{1}_{2}.ckpt".
                               format(args.model_dir, args.run_num, epoch))
                    print("Model Saved")

            # Share weights from evaluation network to target network
            if step % 10000 == 0:
                update_parameters.make(sess)
                logging.info("\nCopied model parameters to target network.")

if __name__ == "__main__":
    log_fmt = '%(asctime)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    logger = logging.getLogger(__name__)

    args = parser.parse_args()
    main(args)

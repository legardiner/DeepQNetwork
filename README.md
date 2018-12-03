# Policy Gradient

Tensorflow implementation of deep Q reinforcment learning for two games from [OpenAI Gym](https://gym.openai.com/): Breakout and Ms Pacman. 

# Getting Started

## Train Models

To run the games with the default hyperparameters, use the following commmands and specify a `run_num` to create a new log directory:

```
python breakout.py --run_num 1
```

```
python mspacman.py --run_num 1
```

The default hyperparameters were selected through experimentation, but can be adjusted by adding arguments to a game launch.

## Visualize Training

To visualize the average total episode reward at each epoch, launch tensorboard with the following command:

```
tensorboard --logdir=breakout/logs/[run_num]
```

## Evaluate Model

To evaluate a model by playing 1000 games and calculating the total reward, run the following command:

```
python breakout_eval.py --model_path path/to/model.ckpt
```

```
python mspacman_eval.py --model_path path/to/model.ckpt
```
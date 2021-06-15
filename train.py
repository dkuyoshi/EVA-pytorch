import argparse

import numpy as np
import torch
import torch.nn as nn

import pfrl
from pfrl import agents, experiments, explorers
from pfrl import nn as pnn
from pfrl import replay_buffers, utils
from pfrl.initializers import init_chainer_default
from pfrl.q_functions import DiscreteActionValueHead, DuelingDQN
from pfrl.wrappers import atari_wrappers

import gym
import gym.wrappers

import json
from q_function import QFunction
from q_function_with_reduction_e import QFunctionR

from value_buffer import ValueBuffer
from eva_replay_buffer import EVAReplayBuffer
from eva import EVA


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='PongNoFrameskip-v4',
                        help='OpenAI Atari domain to perform algorithm on.')
    parser.add_argument('--outdir', type=str, default='results',
                        help='Directory path to save output files.'
                             ' If it does not exist, it will be created.')
    parser.add_argument('--seed', type=int, default=0,
                        help='Random seed [0, 2 ** 31)')
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU to use, set to -1 if no GPU.')
    parser.add_argument('--demo', action='store_true', default=False)
    parser.add_argument('--load', type=str, default=None)
    parser.add_argument('--logging-level', type=int, default=20,
                        help='Logging level. 10:DEBUG, 20:INFO etc.')
    parser.add_argument('--render', action='store_true', default=False,
                        help='Render env states in a GUI window.')
    parser.add_argument('--monitor', action='store_true', default=False,
                        help='Monitor env. Videos and additional information'
                             ' are saved as output files.')
    parser.add_argument('--steps', type=int, default=10 ** 7,
                        help='Total number of timesteps to train the agent.')
    parser.add_argument('--replay-start-size', type=int, default=4 * 10 ** 4,
                        help='Minimum replay buffer size before ' +
                             'performing gradient updates.')
    parser.add_argument('--eval-n-steps', type=int, default=125000)
    parser.add_argument('--eval-interval', type=int, default=250000)
    parser.add_argument('--n-best-episodes', type=int, default=30)
    parser.add_argument('--update_interval', type=int, default=4)
    parser.add_argument('--soft-update-tau', type=float, default=1e-2)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--periodic_steps', type=int, default=20,
                        help='backup insert period')
    parser.add_argument('--value_buffer_neighbors', type=int, default=5,
                        help='Number of k')
    parser.add_argument('--lambdas', type=float, default=0.4,
                        help='Number of λ')
    parser.add_argument('--replay_buffer_neighbors', type=int, default=10,
                        help='Number of M')
    parser.add_argument('--len_trajectory', type=int, default=50,
                        help='max length of trajectory(T)')
    parser.add_argument('--replay_buffer_capacity', type=int, default=500000,
                        help='Replay Buffer Capacity')
    parser.add_argument('--value_buffer_capacity', type=int, default=2000,
                        help='Value Buffer Capacity')
    parser.add_argument('--minibatch_size', type=int, default=48,
                        help='Training batch size')
    parser.add_argument('--target_update_interval', type=int, default=2000,
                        help='Target network period')
    parser.add_argument('--LRU', action='store_true', default=False,
                        help='Use LRU to store in value buffer')
    parser.add_argument('--prioritized_replay', action='store_true', default=False)
    parser.add_argument('--dueling', action='store_true', default=False,
                        help='use dueling dqn')
    parser.add_argument('--noisy_net_sigma', type=float, default=None,
                        help='NoisyNet explorer switch. This disables following options: '
                             '--final-exploration-frames, --final-epsilon, --eval-epsilon')
    parser.add_argument('--num_step_return', type=int, default=1)
    parser.add_argument(
        "--eval-epsilon",
        type=float,
        default=0.001,
        help="Exploration epsilon used during eval episodes.",
    )
    parser.add_argument(
        "--final-epsilon",
        type=float,
        default=0.01,
        help="Final value of epsilon during training.",
    )
    parser.add_argument(
        "--final-exploration-frames",
        type=int,
        default=10 ** 6,
        help="Timesteps after which we stop " + "annealing exploration rate",
    )
    parser.add_argument('--r', action='store_true', default=False,
                        help='Use projection q function')
    parser.add_argument("--lr", type=float, default=2.5e-4, help="Learning rate.")

    args = parser.parse_args()

    import logging
    logging.basicConfig(level=args.logging_level)

    # Set a random seed used in PFRL.
    utils.set_random_seed(args.seed)

    # Set different random seeds for train and test envs.
    train_seed = args.seed
    test_seed = 2 ** 31 - 1 - args.seed

    args.outdir = experiments.prepare_output_dir(
        args, args.outdir,
        time_format='{}/EVA/seed{}/%Y%m%dT%H%M%S.%f'.format(args.env, args.seed))
    print('Output files are saved in {}'.format(args.outdir))

    def make_env(test):
        # Use different random seeds for train and test envs
        env_seed = test_seed if test else train_seed
        env = atari_wrappers.wrap_deepmind(
            atari_wrappers.make_atari(args.env, max_frames=None),
            episode_life=not test,
            clip_rewards=not test)
        env.seed(int(env_seed))
        if test:
            # Randomize actions like epsilon-greedy in evaluation as well
            env = pfrl.wrappers.RandomizeAction(env, args.eval_epsilon)
        if args.monitor:
            env = pfrl.wrappers.Monitor(
                env, args.outdir, mode="evaluation" if test else "training"
            )
        if args.render:
            env = pfrl.wrappers.Render(env)
        return env

    env = make_env(test=False)
    eval_env = make_env(test=True)

    if args.gpu is not None and args.gpu >= 0:
        assert torch.cuda.is_available()
        device = torch.device("cuda:{}".format(args.gpu))
    else:
        device = torch.device("cpu")

    n_actions = env.action_space.n

    n_input = 4

    if args.r:
        q_func = QFunctionR(n_input_channels=n_input, n_actions=n_actions, device=device, )
        rbuf = EVAReplayBuffer(args.replay_buffer_capacity, num_steps=args.num_step_return, key_width=4, device=device,
                               M=args.replay_buffer_neighbors,
                               T=args.len_trajectory)
    else:
        q_func = QFunction(n_input_channels=n_input, n_actions=n_actions, device=device, )
        rbuf = EVAReplayBuffer(args.replay_buffer_capacity, num_steps=args.num_step_return, key_width=256,
                               device=device,
                               M=args.replay_buffer_neighbors,
                               T=args.len_trajectory)

    explorer = explorers.LinearDecayEpsilonGreedy(
        1.0,
        args.final_epsilon,
        args.final_exploration_frames,
        lambda: np.random.randint(n_actions),
    )

    # Draw the computational graph and save it in the output directory.

    # Use the same hyperparameters as the Nature paper
    opt = torch.optim.Adam(q_func.parameters(), lr=args.lr)

    def phi(x):
        # Feature extractor
        return np.asarray(x, dtype=np.float32) / 255

    Agent = EVA

    agent = Agent(q_func, opt, rbuf, gamma=args.gamma,
                  explorer=explorer, gpu=args.gpu, replay_start_size=args.replay_start_size,
                  minibatch_size=args.minibatch_size, update_interval=args.update_interval,
                  target_update_interval=args.target_update_interval, clip_delta=True,
                  phi=phi,
                  target_update_method='hard',
                  soft_update_tau=args.soft_update_tau,
                  n_times_update=1,
                  episodic_update_len=16,
                  len_trajectory=args.len_trajectory,
                  periodic_steps=args.periodic_steps,
                  r=args.r
                  )

    if args.load:
        agent.load(args.load)

    if args.demo:
        eval_stats = experiments.eval_performance(
            env=eval_env,
            agent=agent,
            n_steps=args.eval_n_steps,
            n_episodes=None)
        print('n_episodes: {} mean: {} median: {} stdev {}'.format(
            eval_stats['episodes'],
            eval_stats['mean'],
            eval_stats['median'],
            eval_stats['stdev']))
    else:
        experiments.train_agent_with_evaluation(
            agent=agent, env=env, steps=args.steps,
            eval_n_steps=args.eval_n_steps,
            eval_n_episodes=None,
            eval_interval=args.eval_interval,
            outdir=args.outdir,
            save_best_so_far_agent=True,
            eval_env=eval_env,
        )

if __name__ == '__main__':
    main()


# -*-coding: utf-8 -*-
import argparse
import random

import numpy as np
import gymnasium as gym
import torch
from gymnasium.wrappers import RecordVideo
import torch.autograd as autograd
from BV_Agent.Rainbow import RainbowDQN  # replay memory for rainbow dqn
from config import Env_config, load_ego_agent, Bv_Action
from highway_env.envs.common.action import VehicleAction
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import warnings

warnings.filterwarnings("ignore", category=UserWarning)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Time Retract RL')
    parser.add_argument('--Ego', type=str, default="DQN-Ego", help="the name of Ego model")
    parser.add_argument('--render', action='store_true', help="whether to display during the training")
    parser.add_argument('--train', action='store_true', help="whether to train")
    parser.add_argument('--test', action='store_true', help="whether to test")
    parser.add_argument('--saving', type=int, default=1000, help="saving per episode")
    parser.add_argument('--loading_frame', type=int, default=None, help="load specific trained model")
    parser.add_argument('--lane_count', type=int, default=2, help="the lane_count of the scenario")
    parser.add_argument('--max_train_frame', type=int, default=int(1e4), help="the maxing training frames")
    args = parser.parse_args()
    Ego = args.Ego
    print(f"******* Using {Ego} *******")
    # load specific config
    config = Env_config(Ego)
    config.update({"saving_model_per_frame": args.saving,
                   "lane_count": args.lane_count,
                   "max_train_frame": args.max_train_frame})

    # create the environment
    env = gym.make(config["env_type"], render_mode="rgb_array")
    env.configure({
        "observation": {
            "type": "MultiAgentObservation",  # get the observation from all the controlled vehicle (ego and bvs)
            "observation_config": {
                "type": "Kinematics",  # each vehicle on the road will return its own obs, with their own state in the first row
            }
        },
        "ego_type": config["ego_type"],
        "lanes_count": config["lane_count"],  # the number of the lane
        "vehicles_count": config["vehicle_count"],  # the number of all the vehicle (ego and bvs)
        "controlled_vehicles": config["controlled_vehicle_count"],  # control all the vehicle (ego and bvs), now we control all the vehicle
        "duration": config["simulation_time"],  # simulation time [s]
        "BV_type": config["bv_type"],  # change the bv behavior
        "initial_lane_id": config["lane_count"]-1,  # the ego vehicle will be placed at the bottom lane (lane_id=1 means the top lane)
        "initial_bv_lane_id": config["lane_count"]-2,  # the init selected vehicle placed at different lane compared with ego
        "simulation_frequency": config["simulation_frequency"]
    })
    obs, info = env.reset()

    # load the trained ego agent
    if Ego == "DQN-Ego":
        ego_model = load_ego_agent(ego_model_path=config["Ego_model_path"])  # env can be None if only need prediction from a trained model
        print("Successfully load the trained DQN ego agent")
    else:
        ego_model = None

    # Create bv model
    state_dim = 5 * 5
    action_dim = len(Bv_Action)
    train_seed = random.randint(1, 1000)

    # Train
    if args.train:
        log_dir = f"./AdvLogs/{Ego}-{args.lane_count}lanes"
        writer = SummaryWriter(log_dir=log_dir)
        BV_Agent = RainbowDQN(env=env, memory_size=config["buffer_size"], batch_size=config["batch_size"],
                              target_update=config["update_per_frame"], obs_dim=state_dim,
                              action_dim=action_dim, seed=train_seed)

        print("******* Starting Training *******")
        BV_Agent.train(num_frames=config["max_train_frame"],
                       ego_model=ego_model, writer=writer, args=args, config=config)

    # Test
    if args.test:
        env = RecordVideo(env, video_folder=f"BV_model/{Ego}/{args.lane_count}lanes/videos", episode_trigger=lambda e: True)
        env.unwrapped.set_record_video_wrapper(env)
        env.configure({"simulation_frequency": config["simulation_frequency"]})  # Higher FPS for rendering
        # load the trained bv_model

        BV_Agent = RainbowDQN(env=env, memory_size=config["buffer_size"], batch_size=config["batch_size"],
                              target_update=config["update_per_frame"], obs_dim=state_dim, action_dim=action_dim)
        # load the pretrained bv model
        BV_Agent.load(model_name=Ego, frame=args.loading_frame, lane_count=args.lane_count)
        print("******* Starting Testing *******")
        BV_Agent.test(ego_model=ego_model, args=args, config=config)

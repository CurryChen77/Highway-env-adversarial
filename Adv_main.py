# -*-coding: utf-8 -*-
import gymnasium as gym
from stable_baselines3 import DQN
from collections import namedtuple
import numpy as np
from BV_agent import SACAgent, ReplayBuffer
from highway_env.envs.common.action import VehicleAction

Bv_Action = {
        0: 'LANE_LEFT',
        1: 'IDLE',
        2: 'LANE_RIGHT',
        3: 'FASTER',
        4: 'SLOWER'
    }


def load_ego_agent(ego_model_path, env=None):
    Ego = DQN.load(path=ego_model_path, env=env)
    return Ego


def bvs_init_condition(lanes_count=2, vehicle_count=3):
    bv_init_lane_id = None
    bv_init_speed = None
    bvs_density = None
    return bv_init_lane_id, bv_init_speed, bvs_density


if __name__ == '__main__':
    # Ego Setting
    EGO_MODEL_PATH = "highway_dqn/model"
    VEHICLE_COUNT = 3  # the number of ego and bvs
    LANES_COUNT = 2
    SIMULATION_TIME = 8

    # BV Setting
    MAX_TRAIN_EPISODE = 10000
    BUFFER_SIZE = 100000
    BATCH_SIZE=64

    # Initial condition
    bv_init_lane_id, bv_init_speed, bvs_density = bvs_init_condition(lanes_count=LANES_COUNT, vehicle_count=VEHICLE_COUNT)

    # create the environment
    env = gym.make("highway-adv-v0", render_mode="rgb_array")
    env.configure({
        "observation": {
            "type": "MultiAgentObservation",  # get the observation from all the controlled vehicle (ego and bvs)
            "observation_config": {
                "type": "Kinematics",  # each vehicle on the road will return its own obs, with their own state in the first row
            }
        },
        "lanes_count": LANES_COUNT,  # the number of the lane
        "vehicles_count": VEHICLE_COUNT,  # the number of all the vehicle (ego and bvs)
        "controlled_vehicles": VEHICLE_COUNT,  # control all the vehicle (ego and bvs), now we control all the vehicle
        "duration": SIMULATION_TIME,  # simulation time [s]
        "other_vehicles_type": "highway_env.vehicle.behavior.AdvVehicle",  # change the bv behavior
        "initial_lane_id": LANES_COUNT-1,  # the ego vehicle will be placed at the bottom lane (lane_id=1 means the top lane)
        "bv_init_lane_id": bv_init_lane_id,  # the initial lane id for the bvs to spawn       -> list
        "bv_init_speed": bv_init_speed,  # the initial speed for all the bvs              -> list
        "bvs_density": bvs_density  # the initial spacing density for all the bvs    -> list
    })
    obs, info = env.reset()

    # load the trained ego agent
    ego_model = load_ego_agent(ego_model_path=EGO_MODEL_PATH)  # env can be None if only need prediction from a trained model

    # Create bv model
    obs_shape = env.observation_space[0].shape
    state_dim = obs_shape[0] * obs_shape[1]
    action_dim = len(Bv_Action)
    bv_model = SACAgent(state_dim, action_dim)  # load the bv model
    replay_buffer = ReplayBuffer(BUFFER_SIZE)

    for episode in range(MAX_TRAIN_EPISODE):
        done = truncated = False
        obs, info = env.reset()  # the obs is a tuple containing all the observations of the ego and bvs
        episode_reward = 0

        while not (done or truncated):
            # get ego action
            ego_action = ego_model.predict(obs[0], deterministic=True)[0]  # the first obs is the ego obs
            # get bv action
            bv_action_list = []
            for i in range(len(obs)-1):
                bv_action_idx = np.argmax(bv_model.select_action(obs[i+1]))  # choose the bv observation and convert the obs tu the state
                bv_action = Bv_Action[bv_action_idx]  # bv_action is str type like "LANE_LEFT", "FASTER" and so on
                bv_action_list.append(bv_action)
            # action of all vehicle
            action = VehicleAction(ego_action=ego_action, bv_action_list=bv_action_list)

            # step
            next_obs, reward, done, truncated, info = env.step(action)
            # bv reward
            bv_reward = -1. * reward  # try to minimize the reward of ego car

            # add to the replay buffer
            replay_buffer.add(obs, bv_action, next_obs, reward, done)  # TODO

            # train the bv_model, when the replay got enough data
            if len(replay_buffer.buffer) > BATCH_SIZE:
                bv_model.train(replay_buffer, batch_size=BATCH_SIZE)
            episode_reward += bv_reward

            if done or truncated:
                break
            obs = next_obs

        print(f"Episode: {episode + 1}, Reward: {episode_reward:.2f}")
            # # Render
            # env.render()
    env.close()
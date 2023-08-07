# -*-coding: utf-8 -*-
import gymnasium as gym
from stable_baselines3 import DQN
from collections import namedtuple


Action = namedtuple("Action", ["ego_action", "bv_action_list"])
Bv_action = {
        0: 'LANE_LEFT',
        1: 'IDLE',
        2: 'LANE_RIGHT',
        3: 'FASTER',
        4: 'SLOWER'
    }


def load_ego_agent(ego_model_path, env=None):
    Ego = DQN.load(path=ego_model_path, env=env)
    return Ego


def load_bv_agent():
    pass


if __name__ == '__main__':
    # Ego Setting
    EGO_MODEL_PATH = "highway_dqn/model"
    VEHICLE_COUNT = 3  # the number of ego and bv

    # BV Setting
    MAX_TRAIN_EPISODE = 10000

    # create the environment
    env = gym.make("highway-adv-v0", render_mode="rgb_array")
    env.configure({
        "observation": {
            "type": "MultiAgentObservation",  # get the observation from all the controlled vehicle (ego and bv)
            "observation_config": {
                "type": "Kinematics",
            }
        },
        "lanes_count": 2,  # the number of the lane
        "vehicles_count": VEHICLE_COUNT,  # the number of background vehicle
        "controlled_vehicles": VEHICLE_COUNT,
        "duration": 8,  # [s]
        "other_vehicles_type": "highway_env.vehicle.behavior.AdvVehicle",  # change the bv behavior
        "initial_lane_id": 1  # the lane at the bottom
    })
    obs, info = env.reset()

    # load the trained ego agent
    ego_model = load_ego_agent(ego_model_path=EGO_MODEL_PATH)  # env can be None if only need prediction from a trained model

    # Create bv model
    bv_model = load_bv_agent()  # TODO

    for episode in range(MAX_TRAIN_EPISODE):
        done = truncated = False
        obs, info = env.reset()  # the obs is a tuple containing all the obs from the ego and bv

        while not (done or truncated):
            # the model of the ego vehicle, generate the ego action
            ego_action = ego_model.predict(obs[0], deterministic=True)  # the first obs is the ego obs

            bv_action_list = []
            for i in range(len(obs)-1):
                bv_action_idx = bv_model(obs[i+1])  # choose the bv observation (the first is ego)
                bv_action = Bv_action[bv_action_idx]  # bv_action is str type
                bv_action_list.append(bv_action)
            action = Action(ego_action=ego_action, bv_action_list=bv_action_list)

            # Get ego reward
            next_obs, reward, done, truncated, info = env.step(action)
            bv_reward = -1. * reward  # try to minimize the reward of ego car

            # # Render
            # env.render()
    env.close()
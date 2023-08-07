# -*-coding: utf-8 -*-
import gymnasium as gym
from stable_baselines3 import DQN
from collections import namedtuple


VehicleAction = namedtuple("VehicleAction", ["ego_action", "bv_action_list"])
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


def load_bv_agent():
    pass


if __name__ == '__main__':
    # Ego Setting
    EGO_MODEL_PATH = "highway_dqn/model"
    VEHICLE_COUNT = 3  # the number of ego and bvs
    LANES_COUNT = 2
    SIMULATION_TIME = 8

    # BV Setting
    MAX_TRAIN_EPISODE = 10000

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
        "initial_lane_id": LANES_COUNT-1  # the ego vehicle will be placed at the bottom lane (lane_id=1 means the top lane)
    })
    obs, info = env.reset()

    # load the trained ego agent
    ego_model = load_ego_agent(ego_model_path=EGO_MODEL_PATH)  # env can be None if only need prediction from a trained model

    # Create bv model
    bv_model = load_bv_agent()  # TODO

    for episode in range(MAX_TRAIN_EPISODE):
        done = truncated = False
        obs, info = env.reset()  # the obs is a tuple containing all the observations of the ego and bvs

        while not (done or truncated):
            # the model of the ego vehicle, generate the ego action
            ego_action = ego_model.predict(obs[0], deterministic=True)  # the first obs is the ego obs

            bv_action_list = []
            for i in range(len(obs)-1):
                bv_action_idx = bv_model(obs[i+1])  # choose the bv observation (the first obs is ego, while the rest are bvs)
                bv_action = Bv_Action[bv_action_idx]  # bv_action is str type like "LANE_LEFT", "FASTER" and so on
                bv_action_list.append(bv_action)
            action = VehicleAction(ego_action=ego_action, bv_action_list=bv_action_list)

            # Get ego reward
            next_obs, reward, done, truncated, info = env.step(action)
            bv_reward = -1. * reward  # try to minimize the reward of ego car

            # # Render
            # env.render()
    env.close()
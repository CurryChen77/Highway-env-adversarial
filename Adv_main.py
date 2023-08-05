# -*-coding: utf-8 -*-
# written by chenkeyu
import gymnasium as gym
from gymnasium.wrappers import RecordVideo
from collections import namedtuple

Action = namedtuple("Action", ["ego_action", "bv_action"])
Bv_action = namedtuple("Bv_action", ["adv_acc", "adv_steering"])


def load_ego_agent():
    pass


def load_background_vehicle():
    pass


if __name__ == '__main__':
    # Setting
    MAX_TRAIN_ITERS = 1000

    # create the environment
    env = gym.make("highway-adv-v0", render_mode="rgb_array")
    env.configure({
        "lanes_count": 2,  # the number of the lane
        "vehicles_count": 2,  # the number of background vehicle
        "duration": 6,  # [s]
        "other_vehicles_type": "highway_env.vehicle.behavior.AdvVehicle"  # change the bv behavior
    })
    obs, info = env.reset()
    env = RecordVideo(env, video_folder="highway_adv/videos", episode_trigger=lambda e: True)
    env.unwrapped.set_record_video_wrapper(env)

    # load the trained ego agent
    ego_model = load_ego_agent()  # TODO
    bv_model = load_background_vehicle()  # TODO

    for iter in range(MAX_TRAIN_ITERS):
        done = truncated = False
        obs, info = env.reset()
        while not (done or truncated):
            # the model of the ego vehicle, generate the ego action
            ego_action = ego_model.predict(obs, deterministic=True)
            # the model of the backgound vehicle
            adv_acc, adv_steering = bv_model.predict(obs)
            bv_action = Bv_action(adv_acc=adv_acc, adv_steering=adv_steering)  # the bv_action is also a tuple format
            action = Action(ego_action=ego_action, bv_action=bv_action)

            # Get ego reward
            obs, reward, done, truncated, info = env.step(action)

            # TODO Train the bv model
            # bv_reward = -reward  # try to minimize the reward of ego car

            # Render
            env.render()
    env.close()
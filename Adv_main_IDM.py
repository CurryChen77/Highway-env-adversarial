# -*-coding: utf-8 -*-
import gymnasium as gym
from gymnasium.wrappers import RecordVideo
import numpy as np
from SAC_agent import SACAgent, ReplayBuffer
from highway_env.envs.common.action import VehicleAction
from torch.utils.tensorboard import SummaryWriter

log_dir = "./AdvLogs/IDM-Ego"
writer = SummaryWriter(log_dir=log_dir)

Bv_Action = {
        0: 'LANE_LEFT',
        1: 'IDLE',
        2: 'LANE_RIGHT',
        3: 'FASTER',
        4: 'SLOWER'
    }


if __name__ == '__main__':
    # Ego Setting
    VEHICLE_COUNT = 8  # the number of ego and bvs
    CONTROLLED_VEHICLE_COUNT = 2  # the number of ego and the select bv
    LANES_COUNT = 2
    SIMULATION_TIME = 50

    # BV Setting
    MAX_TRAIN_EPISODE = 100
    TEST_EPISODE = 10
    BUFFER_SIZE = 100000
    BATCH_SIZE = 64
    SAVING_PER_EPISODE = 20
    TRAIN = False
    TEST = True

    # create the environment
    env = gym.make("highway-adv-IDM-v0", render_mode="rgb_array")
    env.configure({
        "observation": {
            "type": "MultiAgentObservation",  # get the observation from all the controlled vehicle (ego and bvs)
            "observation_config": {
                "type": "Kinematics",  # each vehicle on the road will return its own obs, with their own state in the first row
            }
        },
        "lanes_count": LANES_COUNT,  # the number of the lane
        "vehicles_count": VEHICLE_COUNT,  # the number of all the vehicle (ego and bvs)
        "controlled_vehicles": CONTROLLED_VEHICLE_COUNT,  # control all the vehicle (ego and bvs), now we control all the vehicle
        "duration": SIMULATION_TIME,  # simulation time [s]
        "selected_BV_type": "highway_env.vehicle.behavior.AdvVehicle",  # change the bv behavior
        "initial_lane_id": LANES_COUNT-1,  # the ego vehicle will be placed at the bottom lane (lane_id=1 means the top lane)
        "initial_bv_lane_id": LANES_COUNT-2,  # the init selected vehicle placed at different lane compared with ego
        "simulation_frequency": 15
    })
    obs, info = env.reset()

    # Create bv model
    obs_shape = env.observation_space[0].shape
    state_dim = 5 * 5
    action_dim = len(Bv_Action)
    bv_model = SACAgent(state_dim, action_dim)  # load the bv model

    # Training
    if TRAIN:
        replay_buffer = ReplayBuffer(BUFFER_SIZE)
        for episode in range(MAX_TRAIN_EPISODE):
            done = truncated = False
            obs, info = env.reset()  # the obs is a tuple containing all the observations of the ego and bvs
            episode_reward = 0
            while not (done or truncated):
                # get ego action
                ego_action = None
                # get bv action
                bv_original_action = bv_model.select_action(obs[1])  # choose the selected bv observation and convert the obs to the state
                bv_action_idx = np.argmax(bv_original_action)
                bv_action = Bv_Action[bv_action_idx]  # bv_action is str type like "LANE_LEFT", "FASTER" and so on
                # action of all vehicle
                action = VehicleAction(ego_action=ego_action, bv_action=bv_action)
                # step
                obs_list, reward, done, truncated, info = env.step(action)
                # bv reward
                bv_reward = -1. * reward  # try to minimize the reward of ego car and the selected not crashed
                former_obs = obs_list[0]  # the obs from the unchanged selected bv
                updated_obs = obs_list[1]  # the obs from updated selected bv
                # add to the replay buffer
                replay_buffer.add(obs[1], bv_original_action, former_obs[1], reward, done)
                # train the bv_model, when the replay got enough data
                if len(replay_buffer.buffer) > BATCH_SIZE:
                    bv_model.train(replay_buffer, batch_size=BATCH_SIZE)
                episode_reward += bv_reward
                if done or truncated:
                    break
                obs = updated_obs
                # # Render
                env.render()
            # save the model per specific episode
            if episode % SAVING_PER_EPISODE == 0 and episode != 0:
                bv_model.save(model_name="IDM-Ego")
            writer.add_scalar("Reward", episode_reward, episode)
            print(f"Episode: {episode + 1}, Reward: {episode_reward:.2f}")
        env.close()
        writer.close()

    if TEST:
        env = RecordVideo(env, video_folder="BV_model/IDM-Ego/videos", episode_trigger=lambda e: True)
        env.unwrapped.set_record_video_wrapper(env)
        env.configure({"simulation_frequency": 15})  # Higher FPS for rendering
        # load the trained bv_model
        bv_model.load(model_name="IDM-Ego")
        for episode in range(TEST_EPISODE):
            done = truncated = False
            obs, info = env.reset()  # the obs is a tuple containing all the observations of the ego and bvs
            while not (done or truncated):
                # get ego action
                ego_action = None
                # get bv action
                bv_original_action = bv_model.select_action(obs[1])  # choose the selected bv observation and convert the obs to the state
                bv_action_idx = np.argmax(bv_original_action)
                bv_action = Bv_Action[bv_action_idx]  # bv_action is str type like "LANE_LEFT", "FASTER" and so on
                # action of all vehicle
                action = VehicleAction(ego_action=ego_action, bv_action=bv_action)
                # step
                obs_list, reward, done, truncated, info = env.step(action)
                former_obs = obs_list[0]  # the obs from the unchanged selected bv
                updated_obs = obs_list[1]  # the obs from updated selected bv
                if done or truncated:
                    break
                obs = updated_obs
                # # Render
                env.render()
        env.close()

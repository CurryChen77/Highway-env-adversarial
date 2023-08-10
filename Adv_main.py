# -*-coding: utf-8 -*-
import argparse

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
    parser.add_argument('--Ego_model_name', type=str, default="DQN-Ego", help="the name of Ego model")
    parser.add_argument('--render', action='store_true', help="whether to display during the training")
    parser.add_argument('--train', action='store_true', help="whether to display during the training")
    parser.add_argument('--test', action='store_true', help="whether to display during the training")
    args = parser.parse_args()
    Ego_model_name = args.Ego_model_name
    print(f"******* Using {Ego_model_name} *******")
    # load specific config
    config = Env_config(Ego_model_name)

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
    if Ego_model_name == "DQN-Ego":
        ego_model = load_ego_agent(ego_model_path=config["Ego_model_path"])  # env can be None if only need prediction from a trained model
        print("Successfully load the trained DQN ego agent")

    # Create bv model
    state_dim = 5 * 5
    action_dim = len(Bv_Action)
    USE_CUDA = torch.cuda.is_available()

    # Train
    if args.train:
        log_dir = f"./AdvLogs/{Ego_model_name}"
        writer = SummaryWriter(log_dir=log_dir)
        BV_Agent = RainbowDQN(memory_size=config["buffer_size"], batch_size=config["batch_size"],
                                   target_update=config["update_per_episode"], obs_dim=state_dim, action_dim=action_dim)

        print("******* Starting Training *******")
        frame_idx = 0
        update_cnt = 0
        for episode in range(0, config["max_train_episode"]):
            done = truncated = False
            obs, info = env.reset()  # the obs is a tuple containing all the observations of the ego and bvs
            episode_reward = 0
            losses = []
            while not (done or truncated):
                # get ego action
                if Ego_model_name == "DQN-Ego":
                    ego_action = ego_model.predict(obs[0], deterministic=True)[0]  # the first obs is the ego obs
                else:
                    ego_action = None
                # get bv action
                bv_action_idx = BV_Agent.select_action(obs[1].reshape(-1, state_dim))
                bv_action = Bv_Action[int(bv_action_idx)]  # bv_action is str type like "LANE_LEFT", "FASTER" and so on
                # action of all vehicle
                action = VehicleAction(ego_action=ego_action, bv_action=bv_action)
                # step
                obs_list, reward, done, truncated, info = env.step(action)
                # bv reward shaping, original reward -> [0,1]  -reward -> [-1,0] -reward+1 -> [0,1]
                bv_reward = (-1. * reward) + 1
                former_obs = obs_list[0]  # the obs from the unchanged selected bv
                updated_obs = obs_list[1]  # the obs from updated selected bv

                BV_Agent.transition += [bv_reward, former_obs[1].reshape(-1, state_dim), done]

                # N-step transition
                if BV_Agent.use_n_step:
                    one_step_transition = BV_Agent.memory_n.store(*BV_Agent.transition)
                # 1-step transition
                else:
                    one_step_transition = BV_Agent.transition

                # add a single step transition
                if one_step_transition:
                    BV_Agent.memory.store(*one_step_transition)
                # PER: increase beta
                frame_idx += 1
                fraction = min(frame_idx / config["max_train_episode"], 1.0)
                BV_Agent.beta = BV_Agent.beta + fraction * (1.0 - BV_Agent.beta)
                # update the obs
                obs = updated_obs
                episode_reward += bv_reward
                # break
                if done or truncated:
                    break
                # Render
                if args.render:
                    env.render()
                if len(BV_Agent.memory) >= BV_Agent.batch_size:
                    loss = BV_Agent.update_model()
                    losses.append(loss)
                    update_cnt += 1
                    # if hard update is needed
                    if update_cnt % BV_Agent.target_update == 0:
                        BV_Agent._target_hard_update()

            writer.add_scalar("Reward", episode_reward, episode)
            writer.add_scalar("Loss", sum(losses), episode)
            # save the model per specific episode
            if episode % config["saving_model_per_episode"] == 0 and episode != 0:
                BV_Agent.save(model_name=Ego_model_name)
            print(f"Episode: {episode + 1}, Reward: {episode_reward:.2f}, Losses: {sum(losses)}")
        env.close()
        writer.close()
    # Test
    if args.test:
        env = RecordVideo(env, video_folder=f"BV_model/{Ego_model_name}/videos", episode_trigger=lambda e: True)
        env.unwrapped.set_record_video_wrapper(env)
        env.configure({"simulation_frequency": config["simulation_frequency"]})  # Higher FPS for rendering
        # load the trained bv_model

        BV_Agent = RainbowDQN(memory_size=config["buffer_size"], batch_size=config["batch_size"],
                                   target_update=config["update_per_episode"], obs_dim=state_dim, action_dim=action_dim)
        BV_Agent.load(model_name=Ego_model_name)
        print("******* Starting Testing *******")
        for episode in range(config["test_episode"]):
            done = truncated = False
            obs, info = env.reset()  # the obs is a tuple containing all the observations of the ego and bvs
            while not (done or truncated):
                # get ego action
                ego_action = None
                # get bv action
                bv_action_idx = BV_Agent.select_action(obs[1].reshape(-1, state_dim))
                bv_action = Bv_Action[int(bv_action_idx)]  # bv_action is str type like "LANE_LEFT", "FASTER" and so on
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
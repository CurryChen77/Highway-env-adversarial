import gymnasium as gym
from gymnasium.wrappers import RecordVideo
from stable_baselines3 import DQN
from collections import namedtuple

Action = namedtuple("Action", ["ego_action", "bv_action"])

TRAIN = False

if __name__ == '__main__':
    # Create the environment
    env = gym.make("highway-adv-v0", render_mode="rgb_array")
    env.configure({
        "lanes_count": 2,  # the number of the lane
        "vehicles_count": 2,  # the number of background vehicle
        "duration": 8,  # [s]
        "other_vehicles_type": "highway_env.vehicle.behavior.IDMVehicle",  # the behavior of the bv is IDM
        "initial_lane_id": 1
    })
    obs, info = env.reset()

    # Create the model
    model = DQN('MlpPolicy', env,
                policy_kwargs=dict(net_arch=[256, 256]),
                learning_rate=5e-4,
                buffer_size=15000,
                learning_starts=200,
                batch_size=32,
                gamma=0.8,
                train_freq=1,
                gradient_steps=1,
                target_update_interval=50,
                verbose=1,
                tensorboard_log="highway_dqn/")

    # Train the model
    if TRAIN:
        model.learn(total_timesteps=int(2e4))
        model.save("highway_dqn/model")
        del model

    # Run the trained model and record video
    model = DQN.load("highway_dqn/model", env=env)
    env = RecordVideo(env, video_folder="highway_dqn/videos", episode_trigger=lambda e: True)
    env.unwrapped.set_record_video_wrapper(env)
    env.configure({"simulation_frequency": 15})  # Higher FPS for rendering

    for videos in range(10):
        done = truncated = False
        obs, info = env.reset()
        while not (done or truncated):
            # Predict
            action, _states = model.predict(obs, deterministic=True)
            # action = Action(ego_action=action, bv_action=None)
            # Get reward
            obs, reward, done, truncated, info = env.step(action)
            # Render
            env.render()
    env.close()

import argparse
import gymnasium as gym
from gymnasium.wrappers import RecordVideo
from stable_baselines3 import DQN, PPO
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

def get_model_class(model_name):
    model_classes = {
        "DQN": DQN,
        "PPO": PPO,
    }
    return model_classes.get(model_name)


def main(args):
    # Create the environment
    env = gym.make("highway-v0", render_mode="rgb_array")
    env.configure({
        "lanes_count": args.lanes_count,  # the number of the lane
        "vehicles_count": args.vehicles_count,  # the number of background vehicle
        "duration": args.duration,  # [s]
        "other_vehicles_type": "highway_env.vehicle.behavior.IDMVehicle",  # the behavior of the bv is IDM
        "initial_lane_id": 1
    })

    obs, info = env.reset()

    model_type = get_model_class(args.ego_type)

    # Create the model
    print(f">> Ego Model type: {args.ego_type}")
    model = model_type('MlpPolicy', env, verbose=1, tensorboard_log=f"Ego_Agent/Ego_Agent_model/{args.ego_type}/")

    # Train the model
    if args.train:
        print(f">> Starting training {args.ego_type}")
        model.learn(total_timesteps=int(2e4))
        model.save(f"Ego_Agent/Ego_Agent_model/{args.ego_type}/model")
        del model

    if args.test:
        # Run the trained model and record video
        model = model_type.load(f"Ego_Agent/Ego_Agent_model/{args.ego_type}/model", env=env)
        env = RecordVideo(env, video_folder=f"Ego_Agent/Ego_Agent_model/{args.ego_type}/video", episode_trigger=lambda e: True)
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Time Retract RL')
    parser.add_argument('--train', action='store_true', help="whether to train")
    parser.add_argument('--test', action='store_true', help="whether to test")
    parser.add_argument('--vehicles_count', type=int, default=20, help="vehicles_count")
    parser.add_argument('--duration', type=int, default=50, help="duration")
    parser.add_argument('--lanes_count', type=int, default=2, help="lane count")
    parser.add_argument('--ego_type', type=str, default='PPO', help="ego_type")
    args = parser.parse_args()

    main(args)


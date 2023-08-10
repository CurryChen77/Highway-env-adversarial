from stable_baselines3 import DQN


Bv_Action = {
        0: 'LANE_LEFT',
        1: 'IDLE',
        2: 'LANE_RIGHT',
        3: 'FASTER',
        4: 'SLOWER'
    }


def Env_config(Ego_model_name):
    config = {
        "env_type": "highway-adv-v0",
        "bv_type": "highway_env.vehicle.behavior.AdvVehicle",
        "vehicle_count": 12,
        "controlled_vehicle_count": 2,
        "lane_count": 2,
        "simulation_time": 40,
        "max_train_episode": int(1e4),
        "test_episode": 10,
        "buffer_size": int(1e5),
        "batch_size": 64,
        "saving_model_per_episode": 500,
        "num_atoms": 51,
        "v_min": -10,
        "v_max": 10,
        "gamma": 0.99,
        "simulation_frequency": 15,
        "update_per_episode": 100,
        "learning_rate": 0.001,
    }
    if Ego_model_name == "DQN-Ego":
        config.update({
            "Ego_model_path": "highway_dqn/model",
            "ego_type": None
        })
    elif Ego_model_name == "IDM-Ego":
        config.update({
            "ego_type": "highway_env.vehicle.behavior.AggressiveVehicle"
        })
    return config


def load_ego_agent(ego_model_path, env=None):
    Ego = DQN.load(path=ego_model_path, env=env)
    return Ego

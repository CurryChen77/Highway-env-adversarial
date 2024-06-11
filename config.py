from stable_baselines3 import DQN, A2C, PPO


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
        "max_train_frame": int(2e4),
        "test_frame": 10,
        "buffer_size": int(1e5),
        "batch_size": 64,
        "saving_model_per_frame": 5000,
        "num_atoms": 51,
        "v_min": -10,
        "v_max": 10,
        "gamma": 0.99,
        "simulation_frequency": 15,
        "update_per_frame": 100,
        "learning_rate": 1e-4,
    }
    if Ego_model_name == "DQN-Ego":
        config.update({
            "Ego_model_path": "Ego_Agent/Ego_Agent_model/DQN/model",
            "ego_type": None
        })
    elif Ego_model_name == "A2C-Ego":
        config.update({
            "Ego_model_path": "Ego_Agent/Ego_Agent_model/A2C/model",
            "ego_type": None
        })
    elif Ego_model_name == "PPO-Ego":
        config.update({
            "Ego_model_path": "Ego_Agent/Ego_Agent_model/PPO/model",
            "ego_type": None
        })
    elif Ego_model_name == "IDM-Ego":
        config.update({
            "Ego_model_path": None,
            "ego_type": "highway_env.vehicle.behavior.AggressiveVehicle"
        })
    return config


def load_ego_agent(ego_name, ego_model_path=None, env=None):
    if ego_name == "DQN-Ego":
        Ego = DQN.load(path=ego_model_path, env=env)
    elif ego_name == "A2C-Ego":
        Ego = A2C.load(path=ego_model_path, env=env)
    elif ego_name == "PPO-Ego":
        Ego = PPO.load(path=ego_model_path, env=env)
    else:
        Ego = None
    return Ego

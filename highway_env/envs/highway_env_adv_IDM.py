from typing import Dict, Text

import numpy as np

from highway_env import utils
from typing import Optional
from highway_env.envs.highway_env_adv import HighwayEnvAdv
from collections import namedtuple
from highway_env.envs.common.abstract import AbstractEnv
from highway_env.envs.common.action import Action, VehicleAction
from highway_env.road.road import Road, RoadNetwork
from highway_env.utils import near_split
from highway_env.vehicle.controller import ControlledVehicle
from highway_env.vehicle.kinematics import Vehicle

Observation = np.ndarray


class HighwayEnvAdvIDM(HighwayEnvAdv):
    """
    A highway driving environment.

    The vehicle is driving on a straight highway with several lanes, and is rewarded for reaching a high speed,
    staying on the rightmost lanes and avoiding collisions.
    """

    @classmethod
    def default_config(cls) -> dict:
        config = super().default_config()
        config.update({
            "observation": {
                "type": "Kinematics"
            },
            "action": {
                "type": "DiscreteMetaAction",
            },
            "ego_type": "highway_env.vehicle.behavior.IDMVehicle",
            "lanes_count": 4,
            "vehicles_count": 50,
            "controlled_vehicles": 1,
            "initial_lane_id": None,
            "initial_bv_lane_id": None,
            "duration": 40,  # [s]
            "ego_spacing": 2,
            "selected_bv_spacing": 3,
            "vehicles_density": 1,
            "collision_reward": -1,    # The reward received when colliding with a vehicle.
            "right_lane_reward": 0.1,  # The reward received when driving on the right-most lanes, linearly mapped to
                                       # zero for other lanes.
            "high_speed_reward": 0.4,  # The reward received when driving at full speed, linearly mapped to zero for
                                       # lower speeds according to config["reward_speed_range"].
            "lane_change_reward": 0,   # The reward received at each lane change action.
            "reward_speed_range": [20, 30],
            "normalize_reward": True,
            "offroad_terminal": False,
            "selected_BV_type": "highway_env.vehicle.behavior.AdvVehicle",

        })
        return config

    def _create_vehicles(self) -> None:
        """Create some new random vehicles of a given type, and add them on the road."""
        ego_type = utils.class_from_path(self.config["ego_type"])
        other_vehicles_type = utils.class_from_path(self.config["other_vehicles_type"])
        selected_BV_type = utils.class_from_path(self.config["selected_BV_type"])
        # other_per_controlled = near_split(self.config["vehicles_count"], num_bins=self.config["controlled_vehicles"])

        # [2,1] if total 3 vehicle on the road, and the first 2 is one ego and one selected bv, the second 1 is the rest bvs
        other_vehicle_number = self.config["vehicles_count"]-self.config["controlled_vehicles"]
        self.controlled_vehicles = []

        # the controlled vehicle (ego and selected bv)
        # create the ego vehicle (the first vehicle)
        vehicle = Vehicle.create_random(
            self.road,
            speed=25,
            lane_id=self.config["initial_lane_id"],
            spacing=self.config["ego_spacing"]
        )
        # the type of ego vehicle is MDPVehicle
        vehicle = ego_type(self.road, vehicle.position, vehicle.heading, vehicle.speed)
        self.controlled_vehicles.append(vehicle)  # contain the ego and the selected bv for observation
        self.road.vehicles.append(vehicle)

        # create the selected bv (the second controlled)
        selected_bv = selected_BV_type.create_random(
            self.road,
            lane_id=self.config["initial_bv_lane_id"],
            spacing=self.config["selected_bv_spacing"]
        )
        self.selected_bv = selected_bv  # the type of selected bv is AdvVehicle
        self.controlled_vehicles.append(selected_bv)  # contain the ego and the selected bv for observation
        self.road.vehicles.append(selected_bv)

        # create others number of normal bvs
        for _ in range(other_vehicle_number):
            # Create the bv (the rest vehicle)  other_vehicle_type -> IDMVehicle
            vehicle = other_vehicles_type.create_random(
                self.road,
                spacing=1 / self.config["vehicles_density"]
            )
            vehicle.randomize_behavior()
            self.road.vehicles.append(vehicle)

    def _simulate(self, action: Optional[Action] = None) -> None:
        """Perform several steps of simulation with constant action."""
        frames = int(self.config["simulation_frequency"] // self.config["policy_frequency"])
        if type(action) == VehicleAction:
            ego_action = action.ego_action
            bv_action = action.bv_action
        else:
            ego_action = action
            bv_action = None
        for frame in range(frames):
            # Forward action to the ego vehicle
            if ego_action is not None \
                    and not self.config["manual_control"] \
                    and self.steps % int(self.config["simulation_frequency"] // self.config["policy_frequency"]) == 0:
                self.controlled_vehicles[0].act(ego_action)  # self.controlled_vehicle.act() -> MDPVehicle.act(ego_action)

            if bv_action:  # bv_action
                # perform the selected bv action
                self.selected_bv.act(bv_action)  # AdvVehicle.act

            self.road.act()  # self.road.vehicle.act() -> IDMVehicle.act()
            self.road.step(1 / self.config["simulation_frequency"])
            self.steps += 1
            # Automatically render intermediate simulation steps if a viewer has been launched
            # Ignored if the rendering is done offscreen
            if frame < frames - 1:  # Last frame will be rendered through env.render() as usual
                self._automatic_rendering()

        self.enable_auto_render = False

class HighwayEnvAdvIDMFast(HighwayEnvAdvIDM):
    """
    A variant of highway-v0 with faster execution:
        - lower simulation frequency
        - fewer vehicles in the scene (and fewer lanes, shorter episode duration)
        - only check collision of controlled vehicles with others
    """
    @classmethod
    def default_config(cls) -> dict:
        cfg = super().default_config()
        cfg.update({
            "simulation_frequency": 5,
            "lanes_count": 3,
            "vehicles_count": 20,
            "duration": 30,  # [s]
            "ego_spacing": 1.5,
        })
        return cfg

    def _create_vehicles(self) -> None:
        super()._create_vehicles()
        # Disable collision check for uncontrolled vehicles
        for vehicle in self.road.vehicles:
            if vehicle not in self.controlled_vehicles:
                vehicle.check_collisions = False

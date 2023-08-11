from typing import Dict, Text

import numpy as np

from highway_env import utils
from typing import Optional
from highway_env.envs.highway_env import HighwayEnv
from collections import namedtuple
from highway_env.envs.common.abstract import AbstractEnv
from highway_env.envs.common.action import Action, VehicleAction
from highway_env.road.road import Road, RoadNetwork
from highway_env.utils import near_split
from highway_env.vehicle.controller import ControlledVehicle
from highway_env.vehicle.kinematics import Vehicle

Observation = np.ndarray


class HighwayEnvAdv(HighwayEnv):
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
            "ego_type": None,
            "lanes_count": 4,
            "vehicles_count": 50,
            "controlled_vehicles": 1,
            "initial_lane_id": None,
            "initial_bv_lane_id": None,
            "duration": 40,  # [s]
            "ego_spacing": 2,
            "selected_bv_spacing": 3,
            "vehicles_density": 1,
            "collision_reward": -1,      # The reward received when colliding with a vehicle.
            "right_lane_reward": 0.1,    # The reward received when driving on the right-most lanes, linearly mapped to
                                         # zero for other lanes.
            "high_speed_reward": 0.4,    # The reward received when driving at full speed, linearly mapped to zero for
                                         # lower speeds according to config["reward_speed_range"].
            "lane_change_reward": 0,     # The reward received at each lane change action.
            "bv_collision_reward": 0.5,  # The selected bv should not collide with other bv, but collide with ego
            "reward_speed_range": [20, 30],
            "normalize_reward": True,
            "offroad_terminal": False,
            "BV_type": "highway_env.vehicle.behavior.AdvVehicle",

        })
        return config

    def _create_vehicles(self) -> None:
        """Create some new random vehicles of a given type, and add them on the road."""
        bv_type = utils.class_from_path(self.config["BV_type"])
        # other_per_controlled = near_split(self.config["vehicles_count"], num_bins=self.config["controlled_vehicles"])

        other_vehicle_number = self.config["vehicles_count"]-self.config["controlled_vehicles"]  # the rest normal bv
        self.controlled_vehicles = []

        # the controlled vehicle (ego and selected bv)
        # create the ego vehicle (the first vehicle)
        vehicle = Vehicle.create_random(
            self.road,
            speed=25,
            lane_id=self.config["initial_lane_id"],
            spacing=self.config["ego_spacing"]
        )
        if self.config["ego_type"] is not None:
            # the ego car is IDM based
            ego_type = utils.class_from_path(self.config["ego_type"])
            vehicle = ego_type(self.road, vehicle.position, vehicle.heading, vehicle.speed)
        else:
            # the ego car is MDPVehicle
            vehicle = self.action_type.vehicle_class(self.road, vehicle.position, vehicle.heading, vehicle.speed)
            self.action_type.controlled_vehicle = vehicle # set the ego vehicle as the controlled vehicle in action type
        self.controlled_vehicles.append(vehicle)  # contain the ego and the selected bv for observation
        self.road.vehicles.append(vehicle)

        # create the selected bv (the second controlled)
        selected_bv = bv_type.create_random(
            self.road,
            lane_id=self.config["initial_bv_lane_id"],
            spacing=self.config["selected_bv_spacing"]
        )
        selected_bv.selected = True
        self.controlled_vehicles.append(selected_bv)  # contain the ego and the selected bv for observation
        self.road.vehicles.append(selected_bv)

        # create others number of normal bvs
        for _ in range(other_vehicle_number):
            # Create the bv (the rest vehicle)  other_vehicle_type -> IDMVehicle
            vehicle = bv_type.create_random(
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
                if self.config["ego_type"] is not None:
                    self.controlled_vehicles[0].act(ego_action)  # IDM based vehicle act
                else:
                    self.action_type.act(ego_action)  # MDPVehicle act

            if bv_action:  # bv_action
                # perform the selected bv action
                self.controlled_vehicles[1].adv_act(bv_action)  # Adv_act

            self.road.act()  # self.road.vehicle.act() -> IDMVehicle.act()
            self.road.step(1 / self.config["simulation_frequency"])
            self.steps += 1
            # Automatically render intermediate simulation steps if a viewer has been launched
            # Ignored if the rendering is done offscreen
            if frame < frames - 1:  # Last frame will be rendered through env.render() as usual
                self._automatic_rendering()

        self.enable_auto_render = False

    def update_obs(self):
        # updated the selected bv (in the front of ego, and the closest)
        closest_bv = self.road.close_objects_to(
            self.controlled_vehicles[0],  # ego vehicle
            self.PERCEPTION_DISTANCE,
            count=1,  # only need one vehicle
            see_behind=False,  # only looking forward
            sort=True,  # get the closest
            vehicles_only=True
        )
        if closest_bv:
            self.controlled_vehicles[1].selected = False
            self.selected_bv = closest_bv[0]  # closest_bv is a list, the first element is what we want
            self.selected_bv.selected = True
            self.controlled_vehicles[1] = self.selected_bv  # update the second controlled bv (the updated selected bv)
            # After conducing the action, the vehicles' state changes, need to reconsider the new selected bv
            # get the current object need observing (the second car in the agent_observation_type list is selected bv)
            self.observation_type.agents_observation_types[1].observer_vehicle = self.selected_bv


    def _reward(self, action: VehicleAction) -> float:
        """
        The reward is defined to foster driving at high speed, on the rightmost lanes, and to avoid collisions.
        :param action: the last action performed
        :return: the corresponding reward
        """
        ego_action = VehicleAction.ego_action
        bv_action = VehicleAction.bv_action
        rewards = self._rewards(ego_action)  # calculate the reward of the ego agent
        bv_reward = self.bv_reward(bv_action)  # the reward of selected bv
        rewards.update({"bv_collision_reward": bv_reward})
        reward = sum(self.config.get(name, 0) * reward for name, reward in rewards.items())
        if self.config["normalize_reward"]:
            reward = utils.lmap(reward,
                                [self.config["collision_reward"],
                                 self.config["high_speed_reward"] + self.config["right_lane_reward"] + self.config["bv_collision_reward"]],
                                [0, 1])
        reward *= rewards['on_road_reward']  # the reward is with in range [0, 1]

        return reward

    def bv_reward(self, bv_action):
        bv_collision_reward = float(self.controlled_vehicles[1].crashed)  # whether the selected bv has crashed
        # bv_forward_speed = self.controlled_vehicles[1].speed * np.cos(self.controlled_vehicles[1].heading) * -1
        bv_rewards = bv_collision_reward
        return bv_rewards

    def _rewards(self, action: Action) -> Dict[Text, float]:
        # self.vehicle is the first vehicle in controlled_vehicle -> ego vehicle
        neighbours = self.road.network.all_side_lanes(self.vehicle.lane_index)
        lane = self.vehicle.target_lane_index[2] if isinstance(self.vehicle, ControlledVehicle) \
            else self.vehicle.lane_index[2]
        # Use forward speed rather than speed, see https://github.com/eleurent/highway-env/issues/268
        forward_speed = self.vehicle.speed * np.cos(self.vehicle.heading)
        scaled_speed = utils.lmap(forward_speed, self.config["reward_speed_range"], [0, 1])
        return {
            "collision_reward": float(self.vehicle.crashed),
            "right_lane_reward": lane / max(len(neighbours) - 1, 1),
            "high_speed_reward": np.clip(scaled_speed, 0, 1),
            "on_road_reward": float(self.vehicle.on_road)
        }

    def _is_terminated(self) -> bool:
        """The episode is over if the ego vehicle crashed."""
        return (self.vehicle.crashed or
                self.config["offroad_terminal"] and not self.vehicle.on_road)

    def _is_truncated(self) -> bool:
        """The episode is truncated if the time limit is reached."""
        return self.time >= self.config["duration"]

class HighwayEnvAdvFast(HighwayEnvAdv):
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

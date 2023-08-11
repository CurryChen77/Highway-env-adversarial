# Highway-env based adversarial testing
## Visualization
<table>
    <tr>
        <td ><center><img src="image/cut-in.gif">Cut in scenario </center></td>
        <td ><center><img src="image/slow-down.gif">Slow down scenario</center></td>
    </tr>
</table>

## Usage
### Install highway-adv-env
```
python setup.py install
```
### Install the stable-baselines3  
Stable-Baselines3 requires python 3.8+ and PyTorch >= 1.13
* windows:
```
pip install stable-baselines3[extra]
```
* Ubuntu:
```
pip install "stable-baselines3[extra]"
```
### Train the ego agent
```
python .\EgoAgent_trainer_DQN.py
```
### Train the BV model
* DQN-Ego
```
python Adv_main.py --Ego_model_name="DQN-Ego" --train
```
* IDM-EGO
```
python Adv_main.py --Ego_model_name="IDM-Ego" --train
```
### Test the BV model and rendering
* DQN-Ego
```
python Adv_main.py --Ego_model_name="DQN-Ego" --test --render
```
* IDM-EGO
```
python Adv_main.py --Ego_model_name="IDM-Ego"  --test --render
```
### Open the tensorboard
* DQN as ego car
```
tensorboard --logdir=./AdvLogs/DQN-Ego
```
* IDM as ego car
```
tensorboard --logdir=./AdvLogs/IDM-Ego
```
## Environment: highway_env_adv
### Initial condition of all the vehicles：
1. [highway_env_adv](highway_env/envs/highway_env_adv.py) _create_vehicles()function   
* **Step: **
* Put the ego car and the selected bv into self.controlled_vehicles
* The first controlled car is the ego car, while the second is the selected car
* The Ego (MDPVehicle or IDMVehicle type), while all the bvs (AdvVehicle type)
2. Initial place of all the vehicle is decided by their specific speed, land_id, spacing, etc

### 周车行驶行为类：
[AdvVehicle](highway_env/vehicle/behavior.py)
通过改变act函数来确定接收到来自bv_model发出的action动作后，某一周车采取怎样的行为
1. 基于IDM模型的act()，在IDM模型计算得到的acceleration, steering后加上bv_model输出的adv_acc, adv_steering，由Vehicle类（运动学模型）执行
2. high-level的动作，传入字符串动作("FASTER", "LANE_LEFT"等)

### Main脚本
[Adv_main](Adv_main.py)
1. env_config中，obs类型为MultiAgentObservation，从而实现获得的obs为所有车辆独特的obs，其中ego由于在controlled_vehicle中是第一辆，因此
传出的obs中，第一个为ego的obs，其余的为周车的obs
```
"observation": {
   "type": "MultiAgentObservation",  # get the observation from all the controlled vehicle (ego and bv)
   "observation_config": {
       "type": "Kinematics",
   }
},
```
2. bv_model输入选中周车的obs，输出其action
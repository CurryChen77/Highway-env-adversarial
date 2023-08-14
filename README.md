# Highway-env based adversarial testing
## 1. Visualization

### DQN Ego

* 2 lanes

<table>
    <tr>
        <td ><center><img src="image/DQN-Ego-2lanes-brake.gif">Emergency brake </center></td>
        <td ><center><img src="image/DQN-Ego-2lanes-crashed.gif">Rear-end collision</center></td>
    </tr>
</table>

* 3 lanes

<table>
    <tr>
        <td ><center><img src="image/DQN-Ego-3lanes-cut-in1.gif">Cut in</center></td>
        <td ><center><img src="image/DQN-Ego-3lanes-cut-in2.gif">Cut in</center></td>
    </tr>
</table>

### IDM Ego

* 2 lanes

<table>
    <tr>
        <td ><center><img src="image/IDM-Ego-2lanes-crashed.gif">Crashed</center></td>
        <td ><center><img src="image/IDM-Ego-2lanes-cut-in.gif">Cut in</center></td>
    </tr>
</table>

* 3 lanes

<table>
    <tr>
        <td ><center><img src="image/IDM-Ego-3lanes-cut-in.gif">Cut in</center></td>
        <td ><center><img src="image/IDM-Ego-3lanes-surroundcrash.gif">Crashed</center></td>
    </tr>
</table>

## Training losses and reward

* DQN Ego

<table>
    <tr>
        <td ><center><img src="AdvLogs/DQN-Ego-2lanes-10000-losses.png">2 lanes</center></td>
        <td ><center><img src="AdvLogs/DQN-Ego-3lanes-10000-losses.png">3 lanes</center></td>
    </tr>
</table>

* IDM Ego

<table>
    <tr>
        <td ><center><img src="AdvLogs/IDM-Ego-2lanes-10000-losses.png">2 lanes</center></td>
        <td ><center><img src="AdvLogs/IDM-Ego-3lanes-10000-losses.png">3 lanes</center></td>
    </tr>
</table>

## 2. Usage

### 1. Install highway-adv-env
```
python setup.py install
```
### 2. Install the stable-baselines3 (for training the DQN model as Ego) 
Stable-Baselines3 requires python 3.8+ and PyTorch >= 1.13
* windows:
```
pip install stable-baselines3[extra]
```
- Ubuntu:
```
pip install "stable-baselines3[extra]"
```
### 3. Train the ego agent
```
python EgoAgent_trainer_DQN.py
```
### 4. Train the BV model with different Ego agent
- DQN-Ego
```
python Adv_main.py --Ego="DQN-Ego" --train
```
- IDM-EGO
```
python Adv_main.py --Ego="IDM-Ego" --train
```
**default: 2 lanes**

Can be changed by  **--lane_count**

```
python Adv_main.py --Ego="DQN-Ego" --train --lane_count=3
```

### 5. Test the BV model and rendering

- DQN-Ego
```
python Adv_main.py --Ego="DQN-Ego" --test --render
```
- IDM-EGO
```
python Adv_main.py --Ego="IDM-Ego"  --test --render
```
**default: 2 lanes**

Can be changed by  **--lane_count**

```
python Adv_main.py --Ego="DQN-Ego" --test --render --lane_count=3
```

### 6. Open the tensorboard

- DQN as ego car
```
tensorboard --logdir=./AdvLogs/DQN-Ego-3lanes
```
- IDM as ego car
```
tensorboard --logdir=./AdvLogs/IDM-Ego-3lanes
```
## 3. Environment: highway_env_adv
### 1. Initial condition of all the vehicles：
- **`Creating type of all the vehicle`**  
[highway_env_adv](highway_env/envs/highway_env_adv.py) The first controlled car is the ego car (MDPVehicle or IDMVehicle), while the second is the selected car (AdvVehicle)
- **`The initial position of all the vehicle`**  
Initial place of all the vehicle is decided by their specific **speed**, **land_id**, **spacing**, etc

### 2. The AdvVehicle type：
[AdvVehicle](highway_env/vehicle/behavior.py) 
- **`Action`**  
The selected bv is the AdvVehicle type, which using a RL model to perform a high-level action.  
```
ACTIONS_ALL = {
        0: 'LANE_LEFT',
        1: 'IDLE',
        2: 'LANE_RIGHT',
        3: 'FASTER',
        4: 'SLOWER'
    }
```
- **`Observation`**  
The input of the RL model is the corresponding observation of the selected bv
<p align="center">
    <img src="image/obs.png" width="500px"><br/>
    <em>The example observation of ego car.</em>
</p>
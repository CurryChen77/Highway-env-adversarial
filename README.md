# Highway-env based adversarial testing

## Usage
1. install the highway-env-adv
```
python setup.py install
```
2. train the ego agent
```
python .\EgoAgent_trainer_DQN.py
```

## env: highway_env_adv
1. Still need to change the init generation of the background vehicle
2. change the AbstractEnv on the **step** and **_simulate** function to support the action (tuple) for ego and the bv.
But the reward is still the reward for the ego agent, based on the ego_action

## bv behavior type: AdvVehicle
receive the bv_action (tuple) from the bv_model, and add the tiny changes to original IMDVehicle (adv_acc, adv_steering)

## TODO
### 确定自车模型 (pretrained)  
1. 训练ego时，需要确定obs的维度，即所有车辆的数目，不同的车辆数目会影响模型的输入，目前指定为两辆周车，一辆自车

2. 当周车数目增加时，是否需要考虑离自车最近的N辆车作为模型输入的车辆数目保持维度一致，但这需要改变Obs的计算方式，需修改环境中的observation.py文件，不然不能直接用gym的接口
3. 训练自车时，周车策略采用的是IDM

### 确定周车模型 (BV model need training)

#### 周车模型使用方法

1. **针对环境中每一辆车，送入周车模型计算对应单车的action**

   解决方法：采用MultiAgentObservation，每一辆周车都具有其各自的Obs

   且将所有环境车辆都加入到self.controlled_vehicle中，但ego为其中的第一个，且其类别为MDPVehicle，其他的为AdvVehicle

   ![image-20230806153552331](image/obs)

2. 针对环境中所有的车辆，送入周车模型计算所有车辆的action

   obs可保持一致，但输出的action为二维矩阵，行表示每一辆车的action

3. 只针对周车中最近的一车

   obs需要将最近车的state放在第一行，obs需要实时根据最近车辆的改变而改变，输出的action仅一维

#### 周车模型具体动作

1. 以IDM模型为基础，action为连续的acc和steering值，与IDM模型计算结果相加

   出现震荡，转弯和直行时，容易左右震荡，且每做一次决定更新之后，下一个时刻IDM计算时，会将扰动考虑进去？

2. 连续的action值（acc，steering）

   学习困难，因为从控制方面计算，收敛难？

3. **high-level的action决定 (faster, slower, change left, change right, IDLE)**

   周车的网络根据其自己各自的输入Obs，输出high-level action的编号，并重写了_simulate function

#### 周车模型模型训练

1. obs：各个周车各自的obs
2. action：high level action的index
3. reward：自车 -reward

### 确定AdvEnv环境中，车辆的初始条件设置

#### 针对已有创建车辆时的参数

车辆初始条件设置参数：

1. speed，每辆车的初始速度，学习
2. spacing，放置每辆车时，与前车的间隔，学习
3. offset比例，根据前车计算后车时，会在spacing上添加一个随机的偏移量，学习
4. 初始车道，给定
5. 目标车道，随机？

#### 完全自定义

二维矩阵

1. 车辆所处的车道
2. 车辆纵向坐标
3. 车辆初始速度
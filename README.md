# Highway-env based adversarial testing

## env: highway_env_adv
1. Still need to change the init generation of the background vehicle
2. change the AbstractEnv on the **step** and **_simulate** function to support the action (tuple) for ego and the bv.
But the reward is still the reward for the ego agent, based on the ego_action

## bv behavior type: AdvVehicle
receive the bv_action (tuple) from the bv_model, and add the tiny changes to original IMDVehicle (adv_acc, adv_steering)

## TODO
1. specify the ego agent model (pretrained)
2. specify the bv agent model (model need training)
3. specify the create_vehicles function in the AdvEnv
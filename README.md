This repository links a python interface with drone/robotics simulators and real world controllers. The main purpose of this interface is for navigation using neural networks trained using various methods such as imitation and reinforcement learning. It is designed to be customizable, modular, and for tracking/sharing various experiments. 

This repository was released with the publications of _NaviSlim_ at CPS-IoT Week 2024 IoTDI and _NaviSplit_ at WoWMoM 2024. These studies focus on creating adaptable dynamic deep neural networks (ADDNN) for more efficient drone navigation through manipulation of the supporting neural networks. 

**Currently supported simulators:**

Microsoft AirSim

**Currently supported real world controllers:**

DJI Tello

**CITATIONS**

Please cite our NaviSlim paper if using this repository: https://ieeexplore.ieee.org/abstract/document/10562181

_Other related works that use this repository:_

NaviSplit - https://arxiv.org/abs/2406.13086

**AIRSIM SETUP**

step 1: download and install anaconda, then run from anaconda terminal:
conda create --name airsim python=3.12.7

step 2: download repository and run setup.py file. This will create a local folder that is a .gitignore (so feel free to add anything here), along with some sub folders in here for organization.  

step 3: with the new conda environment active, 
conda activate airsim, run the env_airsim.bat file to install all dependecies. Note that a requirements.txt file is not used because the syntax and order of pip installs matter. WARNING: this will force a specific version of pip otherwise the libraries will not install properly!

step 4: download the Blocks AirSim release file from here: https://github.com/microsoft/AirSim/releases. If running on windows, also download and install Microsoft Visual Studio and DirectX. Unpack the downloaded zip file and set the variable at the top of the example_dqn.py file called airsim_release_path, to the file path of the AirSim executable file (.sh for linux and .exe for windows).

step 5: run the python file example_dqn.py to run an example reinforcement learning setup on the Blocks release (make sure the paths are correct), using a flattened depth map as input into an MLP, with the vertical axis locked. It is trained using a DQN reinforcement learning algorithm to get from objective A to B without colliding. 

_bonus steps:_

To run the jupyter notebooks below you will need to create a new conda environment due to dependency issues with AirSim: "conda create --name jupyter python=3.12.7" then execute the env_jupyter.bat file with the new conda env active.

Use the example_test.py file to test a trained model on a set of static test paths we have found using an Astar shortest path algorithm, and the noteboook_eval.ipynb to evaluate and visualize results. You may need to change the variable run_name at the top to direct the test code to which trained config/model you want to test.

Use the example_data.py file to collect any sensor data using rl_drone, and the notebook_data.ipynb to load and visualize the collected data. 

**COMMON ERRORS**

Most of these errors come from AirSim instablities. Since AirSim was depreciated and Microsoft is working on releasing their propietary version, don't count on these issues being resolved any time soon....

AirSim has a dependency issue on tornado. If you created the conda environment properly (see above), you should not run into this issue. However if it is related to tornado or msgpack-rpc then pip install this:
pip install msgpack-rpc-python --upgrade

Occasionally, the drone will become unstable and start oscilating around the z-axis. This is an artificat of AirSim that I mitigated by either teleporting the drone and checking collisions that would have happened on the way, or adding some stability code after each movement that (temporarily) sets the drone velocities to zero. See issue here: https://github.com/microsoft/AirSim/issues/4780

If using moveToPositionAsync() or rotateToPositionAsync(), then AirSim will occassionally freeze and constantly output lookahead errors when colliding with an object. I have tried lowering the timeout (to a few seconds) and setting a lower limit for the distance to both 0.5 and 1 - all to no avail. An alternative solution is to handle these errors with a timer and fidelity test, but this comes with high overhead especially in an iterative reinforcement learning training loop. This is why the current release of rl_drone uses moveByVelocityAsync() and rotateByYawRateAsync() when not using teleport().

Occassionaly, AirSim will crash with an RPCError. This just happens some times when running for a long time. I baked in a crash handler to this repo that will reset the simulator on crash, and reset the episode during DRL training. If the crash handler is not caught, then you will need to manually terminate the program then you can run the .py file again with continue_training=True, to pick up training from the last checkpoint. See issue here: https://github.com/microsoft/AirSim/issues/1757

Excessive lag is likely caused because the simulation is being ran purely on cpu. If this is the case, then you may need to reinstall your nvidia drivers. Otherwise, to run purely on cpu, you can reduce the speedup in the AirSim config file down from the value I typically use of 10.

If you can not succesfully run the conda_env.bat file, it is likely because the conda environment was not properly created.

Malloc issues and crashes when launching airsim likely arise from having an instance of Airsim already running (kill the running instance first).

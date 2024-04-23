**CURRENT SUPPORT**

This repository was released opensource with the release of two conference publications:

T.K. Johnsen,  and M. Levorato, "NaviSlim: Adaptive Context-Aware Navigation and Sensing via Dynamic Slimmable Networks." Proceedings of the 9th ACM/IEEE Conference on Internet of Things Design and Implementation. 2024.

T.K. Johnsen, I. Harshbarger, Z. Xia,  and M. Levorato, "NaviSplit: Dynamic Multi-Branch Split DNNs for Efficient Distributed Autonomous Navigation." 2024 IEEE 25th International Symposium on a World of Wireless, Mobile and Multimedia Networks (WoWMoM). IEEE, 2024.

Patent pending for commerical use.

The current version primarily operates with the opensource GitHub version of the drone simulator Microsoft AirSim: https://github.com/microsoft/AirSim

We also have support for running commands to a real-world drone DJI Tello: https://store.dji.com/shop/tello-series

The current version focuses on running deep reinforcement learning with stable-baselines3: https://stable-baselines3.readthedocs.io/en/master/

The entire repo is heavily modulated and designed for customization, experimentation, and sharing. The example source code focuses on using DRL for navigation, however can be generalized to other applications.

**AIRSIM SETUP**

step 1: download and install anaconda, then run from anaconda terminal: 'conda create --name airsim python=3.10'. https://www.anaconda.com/

step 2: install pytorch with the airsim conda environment active. Our current implementation is proven stable for cuda 11.8. https://pytorch.org/ 

step 3: download repository and run setup.py file: 'python setup.py'. This will create a 'local' folder that is a .gitignore (so feel free to add anything here that you do not want to sync with gihub). It will also create a global_parameters.json file, with settings for the sim - feel free to edit this as appropriate (name your pc for logging purposes and enable/disable rendering graphics to screen). 

step 4: with the airsim conda environment active, then run the "conda_env.bat file", to install all dependecies. Note that a requirements.txt file is not used because the syntax and order of pip installs matter. When running the bat file, you may need to edit the pip install command (but keep the version numbers as those are proven stable).

step 5: download the Blocks AirSim release file from here: https://github.com/microsoft/AirSim/releases. If running on windows, also install directx runtime (https://www.microsoft.com/en-gb/download/details.aspx?id=35) and Microsoft Visual Studio (https://visualstudio.microsoft.com/) specifically development for desktop C++ (community version is fine). Currently proven stable are the maps Blocks and AirSimNH for both windows and linux, and CityEnviron for windows. Move any release zip files to local/airsim_maps and unzip contents directly into this folder (do not create additional sub folders when unzipping). For windows the path should be: rl_drone/local/airsim_maps/Blocks/WindowsNoEditor/Blocks.exe For linux the path should be: rl_drone/local/airsim_maps/LinuxBlocks1.9.1/LinuxNoEditor/Blocks.sh

step 6: run the sample python file "debug.py", to run an AirSim example that connects components needed for basic control and sensor aquisition. Use this file to debug components.

step 7: create a copy of the run_navi.py file and edit how you want! This is also ar unning example, where the default values will use a TD3 DRL algorithm to learn navigation, using the ground truth depth maps returned from AirSim. Happy learning =) 


**ADVANCED**

1. If you want to create a custom component, this repo is specifically designed for that in mind! To insure everything works properly, including serialization of configuration files and debugging capabilities, follow the steps outlined in the code block at the top of the component.py file.

2. AirSim is rather unstable when running for more than an hour or so, which as needed for DRL, especially on linux distros (I have found windows is more stable). Thus there is a crash handling system that is baked into the current implementation. See the goalenv.py file for how it is handled. 


**COMMON ERRORS**

Most of these errors come from AirSim instablities. Since the opensource GitHub AirSim was depreciated and Microsoft is working on releasing their propietary version, I don't count on these issues being resolved any time soon, if ever at all....

1. AirSim has a dependency issue on tornado. If you created the conda environment properly (see above), you should not run into this issue. However if it is related to tornado or msgpack-rpc then pip install like this : 'pip install msgpack-rpc-python --upgrade'

2. Occasionally, the drone will become unstable and start oscilating around the z-axis. I have mitigated this by forcing the velocities to zero after moving. See issue here: https://github.com/microsoft/AirSim/issues/4780

3. If using moveToPositionAsync() or rotateToPositionAsync(), then AirSim will occassionally freeze and constantly output lookahead errors when colliding with an object. This is why the current release of this repo (rl_drone) uses moveByVelocityAsync() and rotateByYawRateAsync().

4. Occassionaly, AirSim will crash with an RPCError. This just happens some times when running for a long time (hours to a day or two). I have added a restart loop to address this. See issue here: https://github.com/microsoft/AirSim/issues/1757

5. Jupyter notebooks do not work from the conda environment due to the tornado dependency.

6. There is possibly a dependency issue with binvox and numpy, if using versions outside of that in the conda_env.bat file -- to fix this, change all np.bool to bool in the binvox source code.

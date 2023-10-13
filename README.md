**AIRSIM SETUP**

step 1: download and install anaconda, then run from anaconda terminal: 'conda create --name airsim python=3.10'

step 2: download repository and run setup.py file: 'python setup.py'. This will create a 'local' folder that is a .gitignore (so feel free to add anything here that you do not want to sync with gihub). It will also create a global_parameters.json file, with settings for the sim - feel free to edit this as appropriate (name your pc for logging purposes and enable/disable rendering graphics to screen). 

step 3: active the new conda environment, 'conda activate airsim', then run the conda_env.bat file, './conda_env.bat' or 'sh ./conda_env.bat', to install all dependecies. Note that a requirements.txt file is not used because the syntax and order of pip installs matter. When running the bat file, you may need to edit the pip install command (but keep the version numbers).

step 4: download the Blocks AirSim release file from here: https://github.com/microsoft/AirSim/releases. If running on windows, also install directx runtime and Microsoft Visual Studio - specifically development for desktop C++. Currently supported is Blocks and AirSimNH for both windows and linux, and CityEnviron for windows. Move any release zip files to local/airsim_maps and unzip contents directly into this folder (do not create additional sub folders when unzipping). For windows the path should be: rl_drone/local/airsim_maps/Blocks/WindowsNoEditor/Blocks.exe For linux the path should be: rl_drone/local/airsim_maps/LinuxBlocks1.9.1/LinuxNoEditor/Blocks.sh

step 5: run the sample python file, 'python debug.py' or 'python3 debug.py', to run an AirSim example that connects components needed for basic control and sensor aquisition. Use this file to debug components.

step 6: create a clone of the run.py file and edit how you want! happy learning =) NOTE: if you want to create a custom component, this repo is specifically designed for that in mind. To insure everything works properly, including serialization of configuration files and debugging capabilities, follow the steps outlined in the code block at the top of the component.py file.


**COMMON ERRORS**

Most of these errors come from AirSim instablities. Since AirSim was depreciated and Microsoft is working on releasing their propietary version, don't count on these issues being resolved any time soon....

1. AirSim has a dependency issue on tornado. If you created the conda environment properly (see above), you should not run into this issue. However if it is related to tornado or msgpack-rpc then pip install this: 'pip install msgpack-rpc-python --upgrade'

2. Occasionally, the drone will become unstable and start oscilating around the z-axis. I have mitigated with this by forcing the velocities to zero after moving. See issue here: https://github.com/microsoft/AirSim/issues/4780

3. If using moveToPositionAsync() or rotateToPositionAsync(), then AirSim will occassionally freeze and constantly output lookahead errors when colliding with an object. This is why the current release of rl_drone uses moveByVelocityAsync() rotateByYawRateAsync().

4. Occassionaly, AirSim will crash with an RPCError. This just happens some times when running for a long time (hours to a day or two). I have added a restart loop to address this. See issue here: https://github.com/microsoft/AirSim/issues/1757

5. Jupyter notebooks do not work from the conda environment due to the tornado dependency.

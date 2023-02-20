**AIRSIM SETUP**

step 1: download repository and run setup.py file. This will create a local folder that is a .gitignore (so feel free to add anything here). It will also create a  global_parameters.json file, edit this as appropriate (optionally name your pc for tracking purposes and enable/disable rendering graphics to screen). 

step 2: download and install anaconda, then run from anaconda terminal:
conda create --name airsim python=3.10
conda activate airsim

step 3: with the new conda environment active, run the conda_env.bat file to install all dependecies. Note a requirements.txt file is not used because the syntax matters.

step 4: download the Blocks AirSim release file from here: https://github.com/microsoft/AirSim/releases. If running on windows, also download and install Microsoft Visual Studio. Currently supported is Blocks and AirSimNH for both windows and linux, and CityEnviron for windows. Move any release zip files to local/airsim_maps and unzip (i.e. after unzipping, the release should have the following strucutre: local/airsim_maps/{release_name}/{os_name}/files).

step 5: run the python file run_drift.py to check your configuration. This should launch Blocks map and make random moves with a quadcopter. This is collecting data related to drone drifts and collision detection accuracy. Let this run for a while. You can copy-paste the drift_eval.py file to the newly created run folder in local/runs then run the notebook at any time to evaluate the data. The drone_eval file will show how much rotational and translational drift the drone is having after issuing commands. You may need to change your setup, such as clock_speed and drone speed to imporve this - to fit your needs. See issues below that address mitigating this.

step 6: run the python file run_train.py to run an example reinforcement learning setup on the Blocks release. This will use a CNN feature extractor to capture depths from the quadcopter camera, using an Atari CNN architecture with randomized weights. Drone dynamics will be fed into an MLP that is added onto the feature extractor. The MLP is then used to decide which actions to take based on the current state. The neural network is trained using a TD3 reinforcement learning algorithm. It is trained to get from objective A to B without colliding. You can copy-paste the train_eval.py file to the newly created run folder in local/runs then run the notebook at any time to evaluate the reinforcement learning algorithm. See paper for more implementation details.

step 7: edit the run_\*.py files as needed to train how you want! happy learning =)
NOTE: if you want to create a custom component, this repo is specifically designed for that in mind. To insure everything works properly, including serialization of configuration files, follow the steps outlined in the code block at the top of the component.py file.


**COMMON ERRORS**

Most of these errors come from AirSim instablities. Since AirSim was depreciated and Microsoft is working on releasing their propietary version, don't count on these issues being resolved any time soon....

1. AirSim has a dependency issue on tornado. If you created the conda environment properly (see above), you should not run into this issue. However if it is related to tornado or msgpack-rpc then pip install this:
pip install msgpack-rpc-python --upgrade

2. Occasionally, the drone will become unstable and start oscilating around the z-axis. This is an artificat of AirSim that I have not been able to mitigate. I have tried moving only along the x-axis, rotating the drone, and playing with PID controller values to no avail. The only real solution to this is to reduce drone speed and add a wait time between moves - which drastically slows down training. See issue here: https://github.com/microsoft/AirSim/issues/4780

3. If using moveToPositionAsync() or rotateToPositionAsync(), then AirSim will occassionally freeze and constantly output lookahead errors when colliding with an object. I have tried lowering the timeout (to a few seconds) and setting a lower limit for the distance to both 0.5 and 1 - all to no avail. An alternative solution is to handle these errors with a timer and fidelity test, but this comes with high overhead especially in an iterative reinforcement learning training loop. This is why the current release of rl_drone uses moveByVelocityAsync() rotateByYawRateAsync().

4. Occassionaly, AirSim will crash with an RPCError. This just happens some times when running for a long time (hours to a day or two). Because of this, you will need to occasionally check that the simulation is running. After a crash, rerun the py file with continue_training=True - this will pick up training from the last checkpoint. See issue here: https://github.com/microsoft/AirSim/issues/1757

5. Jupyter notebooks do not work from the conda environment due to the tornado dependency.

**SETUP**

step 1: 


**COMMON ERRORS**

Most of these errors come from AirSim instablities. Since AirSim was depreciated and Microsoft is working on releasing their propietary version, don't count on these issues being resolved any time soon....

1. AirSim has a dependency issue on tornado. If you created the conda environment properly (see above), you should not run into this issue. However if it is related to tornado or msgpack-rpc then pip install this:
pip install msgpack-rpc-python --upgrade

2. Occasionally, the drone will become unstable and start oscilating around the z-axis. This is an artificat of AirSim that I have not been able to mitigate. I have tried moving only along the x-axis, rotating the drone, and playing with PID controller values to no avail. The only real solution to this is to reduce drone speed and add a wait time between moves - which drastically slows down training. See issue here: https://github.com/microsoft/AirSim/issues/4780

3. If using moveToPositionAsync() or rotateToPositionAsync(), then AirSim will occassionally freeze and constantly output lookahead errors when colliding with an object. I have tried lowering the timeout (to a few seconds) and setting a lower limit for the distance to both 0.5 and 1 - all to no avail. This is why the current release of rl_drone uses moveByVelocityAsync() rotateByYawRateAsync().

4. Occassionaly, AirSim will crash with an RPCError. This just happens some times when running for a long time (hours to a day or two). Because of this, you will need to occasionally check that the simulation is running. After a crash, rerun the py file with continue_training=True - this will pick up training from the last checkpoint. See issue here: https://github.com/microsoft/AirSim/issues/1757

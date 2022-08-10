print('loading libraries...')

# installed pacakages
import os
# local files
from utils import *
from component import *

print('creating components...')
    
# PARAMS (see entire main.py file to set different params)
read_params = False
time_stamp = get_time_stamp()
parameters = {
    'time_stamp':time_stamp,
    'log':True,
    'log_path':os.getcwd() + '/log/',
}
if read_params:
    parameters_path = 'log/parameters.json'
    parameters_read = read_json(parameters_path)
    parameters_read.update(parameters)
    parameters = parameters_read.copy()
    components = deserialize_components(parameters['components'])
else:
    #'''
    map_type = 'AirSim'
    drone_type = 'AirSim'
    sensor_type = 'AirSimCamera'
    transformer_types = ['ResizeImage', 'NormalizeDepth']
    observer_type = 'Single'
    action_type = 'FixedMove'
    move_types = ['left', 'right', 'up', 'down', 'forward', 'backward',
                    'up_left', 'down_right_forward',
                    ]
    actor_type = 'DiscreteActor'
    reward_types = ['Avoid', 'Point']
    rewarder_type = 'Schema'
    terminator_types = ['Collision', 'Point']

    # MAP
    if map_type == 'AirSim':
        from maps.airsimmap import AirSimMap
        map_ = AirSimMap(
            settings=None,
            setting_files=['base'],
            release_file='Blocks',
            release_directory='D:/airsim_releases/',
            settings_directory='maps/airsim_settings/',
        )
    elif map_type == 'UCIField':
        from maps.ucifield import UCIField
        map_ = UCIField(
        )

    # DRONE
    if drone_type == 'AirSim':
        from drones.airsimdrone import AirSimDrone
        drone = AirSimDrone(
        )
    elif drone_type == 'Tello':
        from drones.tello import Tello 
        drone = Tello(
            wifi_name = 'cloud',
            wifi_password = 'bustersword',
        )

    # SENSOR
    if sensor_type == 'AirSimCamera':
        from sensors.airsimcamera import AirSimCamera
        sensor = AirSimCamera(
            camera_view='0',
            image_type=2,
            as_float=True,
            compress=False,
            is_gray=True
        )
    elif sensor_type == 'PortCamera':
        from sensors.portcamera import PortCamera
        sensor = PortCamera(
            port='udp://0.0.0.0:11111',
            is_gray=False,
        )
       
    # TRANSFORMER
    transformer_names=[] # robots in disguise! 
    for transformer_type in transformer_types:
        transformer = None
        if transformer_type == 'ResizeImage':
            from transformers.resizeimage import ResizeImage
            transformer = ResizeImage(
                image_shape=(288, 512)
            )
        elif transformer_type == 'NormalizeDepth':
            from transformers.normalizedepth import NormalizeDepth
            transformer = NormalizeDepth(
                min_depth=0,
                max_depth=100,
            )
        transformer_names.append(transformer._name)

    # OBSERVER
    if observer_type == 'Single':
        from observers.single import Single
        observer = Single(
            sensor_name=sensor._name, 
            transformer_names=transformer_names,
            please_write=True, 
            write_directory='temp/',
        )

    # ACTION
    action_names = []
    if action_type == 'FixedMove':
        from actions.fixedmove import FixedMove 
        for move_type in move_types:
            fixed_move = FixedMove.get_move(
                drone_name=drone._name, 
                move_type=move_type, 
                step_size=5,
                speed=4,
                front_facing=True,
            )
            action_names.append(fixed_move._name)

    # ACTOR
    if actor_type == 'DiscreteActor':
        from actors.discreteactor import DiscreteActor
        actor = DiscreteActor(
            action_names=action_names,
        )

    # REWARD
    reward_names=[]
    for reward_type in reward_types:
        reward = None
        if reward_type == 'Avoid':
            from rewards.avoid import Avoid
            reward = Avoid(
                drone_name = drone._name
            )
        elif reward_type == 'Point':
            from rewards.point import Point
            reward = Point(
                drone_name = drone._name,
                xyz_point = [0, 0, 0],
                min_distance = 5,
                max_distance = 50,
            )
        reward_names.append(reward._name)

    # REWARDER
    if rewarder_type == 'Schema':
        from rewarders.schema import Schema
        rewarder = Schema(
            reward_names=reward_names,
            reward_weights=[4, 2],
        )

    # TERMINATOR
    terminator_names=[]
    for terminator_type in terminator_types:
        terminator = None
        if terminator_type == 'Collision':
            from terminators.collision import Collision
            terminator = Collision(
                drone_name = drone._name
            )
        elif terminator_type == 'Point':
            from terminators.point import Point
            terminator = Point(
                drone_name = drone._name,
                xyz_point = [0, 0, 0],
                min_distance = 5,
            )
        terminator_names.append(terminator._name)

# get all components
components = get_all_components()

# LOG
if parameters['log']:
    if not os.path.exists(parameters['log_path']):
        os.mkdir(parameters['log_path'])
    parameters['components'] = serialize_components(components)
    if read_params:
        write_json(parameters, parameters['log_path'] + 'parameters2.json')
    else:
        write_json(parameters, parameters['log_path'] + 'parameters.json')
 

# ANNNNNDDD WERE OFF!!!!
#try:
print('connecting components...')
for component in components:
    component.connect()


print('testing components...')
for component in components:
    component.test()
    drone.move_to(0, 0, 5, 4, front_facing=False)
    x = input()
    drone.check_collision()

'''
print('running components...')
for component in components:
    component.run()
'''
#except Exception as e:
#    print('EXCEPTION CAUGHT ISSUING STOP COMMAND:', e)
#    for component in components:
#        component.stop()

# report log
components = get_all_components()
for component in reversed(components):
    log_memory(component)
write_json(diary, 'log/diary')

# CLOSE UP SHOP
print('disconnecting components...')
for component in reversed(components):
    component.disconnect()

print('Good bye!')
